#include "visual_frontend.hpp"

VisualFrontend::VisualFrontend(std::shared_ptr<State> state,
                               std::shared_ptr<Frame> frame,
                               std::shared_ptr<MapManager> mapManager,
                               std::shared_ptr<Mapper> mapper,
                               std::shared_ptr<FeatureTracker> featureTracker)
    : state_(state), currFrame_(frame), mapManager_(mapManager),
      mapper_(mapper), featureTracker_(featureTracker)
{
    cv::Size gridSize(state_->imgWidth_ / state_->claheTileSize_, state_->imgHeight_ / state_->claheTileSize_);
    clahe_ = cv::createCLAHE(state_->claheContrastLimit_, gridSize);
}

void VisualFrontend::track(cv::Mat &image, double timestamp)
{
    bool isKeyFrameRequired = process(image, timestamp);

    if (isKeyFrameRequired)
    {
        mapManager_->createKeyframe(currImage_, image);
        if (!state_->slamResetRequested_ && state_->slamReadyForInit_)
        {
            Keyframe kf(currFrame_->keyframeId_, image);
            mapper_->processNewKeyframe(kf);
        }
    }
}

bool VisualFrontend::process(cv::Mat &image, double timestamp)
{
    preprocessImage(image);

    if (currFrame_->id_ == 0)
        return true;

    // Pose prediction
    Sophus::SE3d Twc = currFrame_->getTwc();
    motionModel_.applyMotionModel(Twc, timestamp);
    currFrame_->setTwc(Twc);

    kltTrackingFromMotionPrior();

    if (!state_->slamReadyForInit_)
    {
        if (currFrame_->numKeypoints2d_ < 50)
        {
            state_->slamResetRequested_ = true;
            return false;
        }
        if (checkReadyForInit())
        {
            state_->slamReadyForInit_ = true;
            return true;
        }
        if (state_->debug_)
            ; // logs omitidos para WASM, puede habilitarse con macro
        return false;
    }
    else
    {
        bool success = computePose();
        if (!success)
        {
            poseFailedCounter_++;
            if (poseFailedCounter_ > 3)
            {
                state_->slamResetRequested_ = true;
                return false;
            }
        }
        else
        {
            poseFailedCounter_ = 0;
        }
        motionModel_.updateMotionModel(currFrame_->Twc_, timestamp);
        return checkNewKeyframeRequired();
    }
}

void VisualFrontend::kltTrackingFromMotionPrior()
{
    std::vector<int> v3dkpids, vkpids;
    std::vector<cv::Point2f> v3dkps, v3dpriors, vkps, vpriors;
    std::vector<bool> vkpis3d;
    v3dkpids.reserve(currFrame_->numKeypoints3d_);
    v3dkps.reserve(currFrame_->numKeypoints3d_);
    v3dpriors.reserve(currFrame_->numKeypoints3d_);
    vkpids.reserve(currFrame_->numKeypoints_);
    vkps.reserve(currFrame_->numKeypoints_);
    vpriors.reserve(currFrame_->numKeypoints_);
    vkpis3d.reserve(currFrame_->numKeypoints_);

    for (const auto &it: currFrame_->mapKeypoints_)
    {
        const auto &keypoint = it.second;
        if (state_->kltUsePrior_ && keypoint.is3d_)
        {
            auto mapPointsIt = mapManager_->mapMapPoints_.find(keypoint.keypointId_);
            if (mapPointsIt != mapManager_->mapMapPoints_.end())
            {
                auto projpx = currFrame_->projWorldToImageDist(mapPointsIt->second->getPoint());
                if (currFrame_->isInImage(projpx))
                {
                    v3dkps.push_back(keypoint.px_);
                    v3dpriors.push_back(projpx);
                    v3dkpids.push_back(keypoint.keypointId_);
                    vkpis3d.push_back(true);
                    continue;
                }
            }
        }
        vkpids.push_back(keypoint.keypointId_);
        vkps.push_back(keypoint.px_);
        vpriors.push_back(keypoint.px_);
    }

    if (state_->kltUsePrior_ && !v3dpriors.empty())
    {
        std::vector<bool> keypointStatus;
        featureTracker_->fbKltTracking(
                prevPyramid_, currPyramid_, state_->kltWinSizeWH_, 1,
                state_->kltError_, state_->kltMaxFbDistance_,
                v3dkps, v3dpriors, keypointStatus);

        size_t numGood = 0, numKeypoints = v3dkps.size();
        for (size_t i = 0; i < numKeypoints; i++)
        {
            if (keypointStatus[i])
            {
                currFrame_->updateKeypoint(v3dkpids[i], v3dpriors[i]);
                numGood++;
            }
            else
            {
                vkpids.push_back(v3dkpids[i]);
                vkps.push_back(v3dkps[i]);
                vpriors.push_back(v3dpriors[i]);
            }
        }

        if (numGood < 0.33 * numKeypoints)
        {
            p3pReq_ = true;
            vpriors = vkps;
        }
    }

    if (!vkps.empty())
    {
        std::vector<bool> keypointStatus;
        featureTracker_->fbKltTracking(
                prevPyramid_, currPyramid_, state_->kltWinSizeWH_, state_->kltPyramidLevels_,
                state_->kltError_, state_->kltMaxFbDistance_,
                vkps, vpriors, keypointStatus);

        size_t numGood = 0, numKeypoints = vkps.size();
        for (size_t i = 0; i < numKeypoints; i++)
        {
            if (keypointStatus[i])
            {
                currFrame_->updateKeypoint(vkpids[i], vpriors[i]);
                numGood++;
            }
            else
            {
                mapManager_->removeObsFromCurrFrameById(vkpids[i]);
            }
        }
    }
}

bool VisualFrontend::computePose()
{
    size_t num3dKeypoints = currFrame_->numKeypoints3d_;
    if (num3dKeypoints < 4) return false;

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vbvs, vwpts;
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> vkps;
    std::vector<int> vkpids, outliersIndices;
    vbvs.reserve(num3dKeypoints); vwpts.reserve(num3dKeypoints); vkpids.reserve(num3dKeypoints); outliersIndices.reserve(num3dKeypoints); vkps.reserve(num3dKeypoints);

    bool doP3P = p3pReq_ || state_->p3pEnabled_;
    for (const auto &it: currFrame_->mapKeypoints_)
    {
        if (!it.second.is3d_) continue;
        const auto &kp = it.second;
        auto mapPointsIt = mapManager_->mapMapPoints_.find(kp.keypointId_);
        if (mapPointsIt == mapManager_->mapMapPoints_.end() || mapPointsIt->second == nullptr) continue;
        if (doP3P) vbvs.push_back(kp.bv_);
        vkps.push_back(Eigen::Vector2d(kp.unpx_.x, kp.unpx_.y));
        vwpts.push_back(mapPointsIt->second->getPoint());
        vkpids.push_back(kp.keypointId_);
    }

    Sophus::SE3d Twc = currFrame_->getTwc();
    bool doOptimize = false, success = false;

    if (doP3P)
    {
        success = MultiViewGeometry::p3pRansac(
            vbvs, vwpts, state_->multiViewRansacNumIterations_, state_->multiViewRansacError_,
            doOptimize, state_->multiViewRandomEnabled_,
            currFrame_->cameraCalibration_->fx_, currFrame_->cameraCalibration_->fy_,
            Twc, outliersIndices);

        size_t numInliers = vwpts.size() - outliersIndices.size();
        if (!success || numInliers < 5 || Twc.translation().array().isInf().any() || Twc.translation().array().isNaN().any())
        {
            resetFrame();
            return false;
        }
        currFrame_->setTwc(Twc);
        int k = 0;
        for (const auto &index : outliersIndices)
        {
            mapManager_->removeObsFromCurrFrameById(vkpids[index - k]);
            vkps.erase(vkps.begin() + index - k);
            vwpts.erase(vwpts.begin() + index - k);
            vkpids.erase(vkpids.begin() + index - k);
            k++;
        }
        outliersIndices.clear();
    }

    // PnP refinement
    bool useRobust = true;
    size_t maxIterations = 5;
    success = MultiViewGeometry::ceresPnP(
        vkps, vwpts, Twc, maxIterations, state_->robustCostThreshold_, useRobust, state_->robustCostRefineWithL2_,
        currFrame_->cameraCalibration_->fx_, currFrame_->cameraCalibration_->fy_,
        currFrame_->cameraCalibration_->cx_, currFrame_->cameraCalibration_->cy_, outliersIndices);

    size_t numInliers = vwpts.size() - outliersIndices.size();
    if (!success || numInliers < 5 || outliersIndices.size() > 0.5 * vwpts.size() ||
        Twc.translation().array().isInf().any() || Twc.translation().array().isNaN().any())
    {
        if (!doP3P)
            p3pReq_ = true;
        resetFrame();
        return false;
    }

    currFrame_->setTwc(Twc);
    p3pReq_ = false;
    for (const auto &idx : outliersIndices)
        mapManager_->removeObsFromCurrFrameById(vkpids[idx]);
    return true;
}

bool VisualFrontend::checkReadyForInit()
{
    double avgComputedRotParallax = computeParallax(currFrame_->keyframeId_, false, true);
    if (avgComputedRotParallax <= state_->minAvgRotationParallax_) return false;

    auto keyframeIt = mapManager_->mapKeyframes_.find(currFrame_->keyframeId_);
    if (keyframeIt == mapManager_->mapKeyframes_.end() || !keyframeIt->second) return false;
    auto prevKeyframe = keyframeIt->second;

    size_t numKeypoints = currFrame_->numKeypoints_;
    if (numKeypoints < 8) return false;

    std::vector<int> keypointIds, outliersIndices;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> vkfbvs, vcurbvs;
    keypointIds.reserve(numKeypoints); outliersIndices.reserve(numKeypoints);
    vkfbvs.reserve(numKeypoints); vcurbvs.reserve(numKeypoints);

    Eigen::Matrix3d RCurrKeyframe = prevKeyframe->getTcw().rotationMatrix() * currFrame_->getTwc().rotationMatrix();
    int numParallax = 0;
    float avgRotParallax = 0.f;
    std::vector<float> parallaxVec;

    for (const auto &it: currFrame_->mapKeypoints_)
    {
        const auto &keypoint = it.second;
        auto keyframeKeypoint = prevKeyframe->getKeypointById(keypoint.keypointId_);
        if (keyframeKeypoint.keypointId_ != keypoint.keypointId_) continue;
        vkfbvs.push_back(keyframeKeypoint.bv_);
        vcurbvs.push_back(keypoint.bv_);
        keypointIds.push_back(keypoint.keypointId_);

        Eigen::Vector3d rotBv = RCurrKeyframe * keypoint.bv_;
        Eigen::Vector3d unpx = currFrame_->cameraCalibration_->K_ * rotBv;
        cv::Point2f rotpx(unpx.x() / unpx.z(), unpx.y() / unpx.z());

        float parallax = cv::norm(rotpx - keyframeKeypoint.unpx_);
        avgRotParallax += parallax;
        parallaxVec.push_back(parallax);
        numParallax++;
    }

    if (numParallax < 8) return false;
    avgRotParallax /= float(numParallax);

    if (avgRotParallax < state_->minAvgRotationParallax_) return false;

    Eigen::Matrix3d Rwc;
    Eigen::Vector3d twc;
    Rwc.setIdentity();
    twc.setZero();

    bool success = MultiViewGeometry::compute5ptEssentialMatrix(
        vkfbvs, vcurbvs, state_->multiViewRansacNumIterations_, state_->multiViewRansacError_,
        true, state_->multiViewRandomEnabled_, currFrame_->cameraCalibration_->fx_,
        currFrame_->cameraCalibration_->fy_, Rwc, twc, outliersIndices);

    if (!success) return false;

    for (const auto &index : outliersIndices)
        mapManager_->removeObsFromCurrFrameById(keypointIds[index]);

    twc.normalize();
    currFrame_->setTwc(Rwc, twc);

    return true;
}

bool VisualFrontend::checkNewKeyframeRequired()
{
    auto keyframeIt = mapManager_->mapKeyframes_.find(currFrame_->keyframeId_);
    if (keyframeIt == mapManager_->mapKeyframes_.end()) return false;
    auto keyframe = keyframeIt->second;

    double medianRotParallax = computeParallax(keyframe->keyframeId_, true, true);
    int idDiff = currFrame_->id_ - keyframe->id_;

    if (idDiff >= 5 && currFrame_->numOccupiedCells_ < 0.33 * state_->frameMaxNumKeypoints_) return true;
    if (idDiff >= 2 && currFrame_->numKeypoints3d_ < 20) return true;
    if (idDiff < 2 && currFrame_->numKeypoints3d_ > 0.5 * state_->frameMaxNumKeypoints_) return false;

    bool cx = medianRotParallax >= state_->minAvgRotationParallax_ / 2.;
    bool c0 = medianRotParallax >= state_->minAvgRotationParallax_;
    bool c1 = currFrame_->numKeypoints3d_ < 0.75 * keyframe->numKeypoints3d_;
    bool c2 = currFrame_->numOccupiedCells_ < 0.5 * state_->frameMaxNumKeypoints_ && currFrame_->numKeypoints3d_ < 0.85 * keyframe->numKeypoints3d_;

    return (c0 || c1 || c2) && cx;
}

float VisualFrontend::computeParallax(const int keyframeId, bool doUnRotate, bool doMedian)
{
    auto keyframeIt = mapManager_->mapKeyframes_.find(keyframeId);
    if (keyframeIt == mapManager_->mapKeyframes_.end()) return 0.f;

    Eigen::Matrix3d Rkfcur(Eigen::Matrix3d::Identity());
    if (doUnRotate)
    {
        Eigen::Matrix3d Rkfw = keyframeIt->second->getRcw();
        Eigen::Matrix3d Rwcur = currFrame_->getRwc();
        Rkfcur = Rkfw * Rwcur;
    }

    float avgParallax = 0.f;
    int numParallax = 0;
    std::vector<float> parallaxVec;

    for (const auto &it: currFrame_->mapKeypoints_)
    {
        const auto &kp = it.second;
        auto keypoint = keyframeIt->second->getKeypointById(kp.keypointId_);
        if (keypoint.keypointId_ != kp.keypointId_) continue;
        cv::Point2f unpx = kp.unpx_;
        if (doUnRotate)
            unpx = keyframeIt->second->projCamToImage(Rkfcur * kp.bv_);
        float parallax = cv::norm(unpx - keypoint.unpx_);
        avgParallax += parallax;
        parallaxVec.push_back(parallax);
        numParallax++;
    }

    if (numParallax == 0) return 0.f;
    avgParallax /= float(numParallax);

    if (doMedian)
    {
        size_t mid = parallaxVec.size() / 2;
        std::nth_element(parallaxVec.begin(), parallaxVec.begin() + mid, parallaxVec.end());
        avgParallax = parallaxVec[mid];
    }
    return avgParallax;
}

void VisualFrontend::preprocessImage(cv::Mat &image)
{
    cv::swap(currImage_, prevImage_);
    if (state_->claheEnabled_)
        clahe_->apply(image, currImage_);
    else
        currImage_ = image;

    if (state_->kltEnabled_)
    {
        if (!currPyramid_.empty())
            prevPyramid_.swap(currPyramid_);
        cv::buildOpticalFlowPyramid(currImage_, currPyramid_, state_->kltWinSize_, state_->kltPyramidLevels_);
    }
}

void VisualFrontend::resetFrame()
{
    auto mapKeypoints = currFrame_->mapKeypoints_;
    for (const auto &keypoint : mapKeypoints)
        mapManager_->removeObsFromCurrFrameById(keypoint.first);
    currFrame_->mapKeypoints_.clear();
    currFrame_->gridKeypointsIds_.clear();
    currFrame_->gridKeypointsIds_.resize(currFrame_->gridCells_);
    currFrame_->numKeypoints_ = 0;
    currFrame_->numKeypoints2d_ = 0;
    currFrame_->numKeypoints3d_ = 0;
    currFrame_->numOccupiedCells_ = 0;
}

void VisualFrontend::reset()
{
    currImage_.release();
    prevImage_.release();
    currPyramid_.clear();
    prevPyramid_.clear();
    keyframePyramid_.clear();
    poseFailedCounter_ = 0;
}