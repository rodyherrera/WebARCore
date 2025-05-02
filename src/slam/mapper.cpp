#include "mapper.hpp"
#include <memory>
#include <algorithm>

Mapper::Mapper(std::shared_ptr<State> state, std::shared_ptr<MapManager> mapManager, std::shared_ptr<Frame> frame)
        : state_(std::move(state)), mapManager_(std::move(mapManager)), currFrame_(std::move(frame)), 
          optimizer_(std::make_unique<Optimizer>(state_, mapManager_))
{
}

void Mapper::processNewKeyframe(const Keyframe &keyframe)
{
    std::shared_ptr<Frame> newKeyframe = mapManager_->getKeyframe(keyframe.keyframeId_);
    assert(newKeyframe);

    // just keep the last 30 keyframes in our map
    if (keyframe.keyframeId_ > 30)
    {
        mapManager_->removeKeyframe(keyframe.keyframeId_ - 30);
    }

    // Triangulate temporal
    if (newKeyframe->keyframeId_ > 0 && newKeyframe->numKeypoints2d_ > 0)
    {
        triangulateTemporal(*newKeyframe);
    }

    // check if reset is required
    if (state_->slamReadyForInit_)
    {
        // Check for reset conditions
        if ((keyframe.keyframeId_ == 1 && newKeyframe->numKeypoints3d_ < 30) || 
            (keyframe.keyframeId_ < 10 && newKeyframe->numKeypoints3d_ < 3))
        {
            if (state_->debug_)
            {
                std::cout << "- [Mapper]: NewKeyframe - Reset Requested. " 
                          << (keyframe.keyframeId_ == 1 ? "Bad initialization detected! " : 
                             "Num 3D kps:" + std::to_string(newKeyframe->numKeypoints3d_)) << std::endl;
            }

            state_->slamResetRequested_ = true;
            return;
        }
    }

    // Update the map points and the covisible graph between keyframes
    mapManager_->updateFrameCovisibility(*newKeyframe);

    currFrame_->covisibleKeyframeIds_ = newKeyframe->covisibleKeyframeIds_;

    if (keyframe.keyframeId_ > 0)
    {
        matchingToLocalMap(*newKeyframe);
    }

    // do bundle adjustment and map filtering
    optimize(newKeyframe);
}

void Mapper::optimize(const std::shared_ptr<Frame> &keyframe)
{
    // apply local BA
    if (keyframe->keyframeId_ >= 2 && keyframe->numKeypoints3d_ != 0)
    {
        optimizer_->localBA(*keyframe);
    }

    // apply map filtering
    if (state_->mapKeyframeFilteringRatio_ < 1.0 && keyframe->keyframeId_ >= 20)
    {
        const auto covisibleKeyframeMap = keyframe->getCovisibleKeyframeMap();
        const float filteringRatio = state_->mapKeyframeFilteringRatio_;
        const int minObservations = state_->baMinNumCommonKeypointsObservations_ / 2;

        for (auto it = covisibleKeyframeMap.rbegin(); it != covisibleKeyframeMap.rend(); it++)
        {
            int keyframeId = it->first;

            if (keyframeId == 0)
            {
                break;
            }

            if (keyframeId >= keyframe->keyframeId_)
            {
                continue;
            }

            auto coKeyframe = mapManager_->getKeyframe(keyframeId);
            if (coKeyframe == nullptr)
            {
                keyframe->removeCovisibleKeyframe(keyframeId);
                continue;
            }
            else if ((int) coKeyframe->numKeypoints3d_ < minObservations)
            {
                mapManager_->removeKeyframe(keyframeId);
                continue;
            }

            size_t numGoodObservations = 0;
            size_t numTotal = 0;

            for (const auto &kp: coKeyframe->getKeypoints3d())
            {
                auto mapPoint = mapManager_->getMapPoint(kp.keypointId_);

                if (mapPoint == nullptr)
                {
                    mapManager_->removeMapPointObs(kp.keypointId_, keyframeId);
                    continue;
                }
                else if (mapPoint->isBad())
                {
                    continue;
                }
                else if (mapPoint->getObservedKeyframeIds().size() > 4)
                {
                    numGoodObservations++;
                }

                numTotal++;
            }

            if (numTotal > 0 && static_cast<float>(numGoodObservations) / numTotal > filteringRatio)
            {
                mapManager_->removeKeyframe(keyframeId);
            }
        }
    }
}

void Mapper::triangulateTemporal(Frame &frame)
{
    const std::vector<Keypoint> &keypoints = frame.getKeypoints2d();

    if (keypoints.empty())
    {
        return;
    }

    const Sophus::SE3d Twcj = frame.getTwc();
    const size_t numKeypoints = keypoints.size();

    // Use a single keyframe pointer and reset it when needed
    std::shared_ptr<Frame> keyframe;
    
    // Cache for relative motion data
    int relKeyframeId = -1;
    Sophus::SE3d Tcicj;
    Sophus::SE3d Tcjci;
    Eigen::Matrix3d Rcicj;

    int good = 0;
    int candidates = 0;

    const float maxReprojError = state_->mapMaxReprojectionError_;

    // Avoid repeated allocations in the loop
    for (size_t i = 0; i < numKeypoints; i++)
    {
        const auto &keypoint = keypoints[i];
        
        // Get related map point and check if it is ready to be triangulated
        std::shared_ptr<MapPoint> mapPoint = mapManager_->getMapPoint(keypoint.keypointId_);

        if (!mapPoint)
        {
            mapManager_->removeMapPointObs(keypoint.keypointId_, frame.keyframeId_);
            continue;
        }

        // If map point is already 3D continue (should not happen)
        if (mapPoint->is3d_)
        {
            continue;
        }

        // Get the set of keyframes sharing observation of this 2D map point
        const std::set<int> &coKeyframeIds = mapPoint->getObservedKeyframeIds();

        // Continue if new keyframe is the only one observing it
        if (coKeyframeIds.size() < 2)
        {
            continue;
        }

        int keyframeId = *coKeyframeIds.begin();

        if (frame.keyframeId_ == keyframeId)
        {
            continue;
        }

        // Get the 1st keyframe observation of the related map point
        keyframe = mapManager_->getKeyframe(keyframeId);

        if (!keyframe)
        {
            continue;
        }

        // Compute relative motion between new keyframe and selected keyframe (only if req.)
        if (relKeyframeId != keyframeId)
        {
            Sophus::SE3d Tciw = keyframe->getTcw();
            Tcicj = Tciw * Twcj;
            Tcjci = Tcicj.inverse();
            Rcicj = Tcicj.rotationMatrix();

            relKeyframeId = keyframeId;
        }

        const Keypoint &keyframeKeypoint = keyframe->getKeypointById(keypoint.keypointId_);

        if (keyframeKeypoint.keypointId_ != keypoint.keypointId_)
        {
            continue;
        }

        // Check rotation-compensated parallax
        cv::Point2f rotPx = frame.projCamToImage(Rcicj * keypoint.bv_);
        double parallax = cv::norm(keyframeKeypoint.unpx_ - rotPx);

        candidates++;

        // Compute 3D pos and check if its good or not
        Eigen::Vector3d lPoint = MultiViewGeometry::triangulate(Tcicj, keyframeKeypoint.bv_, keypoint.bv_);

        // Project into right cam (new keyframe)
        Eigen::Vector3d rPoint = Tcjci * lPoint;

        // Ensure that the 3D map point is in front of both camera
        if (lPoint.z() < 0.1 || rPoint.z() < 0.1)
        {
            if (parallax > 20.0)
            {
                mapManager_->removeMapPointObs(keyframeKeypoint.keypointId_, frame.keyframeId_);
            }
            continue;
        }

        // Remove map point with high reprojection error
        cv::Point2f lPxProj = keyframe->projCamToImage(lPoint);
        cv::Point2f rPxProj = frame.projCamToImage(rPoint);
        float lDist = cv::norm(lPxProj - keyframeKeypoint.unpx_);
        float rDist = cv::norm(rPxProj - keypoint.unpx_);

        if (lDist > maxReprojError || rDist > maxReprojError)
        {
            if (parallax > 20.0)
            {
                mapManager_->removeMapPointObs(keyframeKeypoint.keypointId_, frame.keyframeId_);
            }
            continue;
        }

        // The 3D pos is good, update SLAM map point and related keyframe / Frame
        Eigen::Vector3d wpt = keyframe->projCamToWorld(lPoint);
        mapManager_->updateMapPoint(keypoint.keypointId_, wpt, 1.0 / lPoint.z());

        good++;
    }
}

float Mapper::computeFOVThreshold(const Frame &frame) const
{
    const float fovV = 0.5 * frame.cameraCalibration_->imgHeight_ / frame.cameraCalibration_->fy_;
    const float fovH = 0.5 * frame.cameraCalibration_->imgWidth_ / frame.cameraCalibration_->fx_;

    const float maxRadFov = std::atan(fovH > fovV ? fovH : fovV);
    return std::cos(maxRadFov);
}

bool Mapper::matchingToLocalMap(Frame &frame)
{
    // Maximum number of map points to track
    const size_t maxNumLocalMapPoints = state_->frameMaxNumKeypoints_ * 10;
    const size_t halfMaxPoints = maxNumLocalMapPoints / 2;

    // get local map of oldest co-keyframe and add it to set of map points to search for
    const auto &covisibleKeyframeMap = frame.getCovisibleKeyframeMap();

    if (!covisibleKeyframeMap.empty() && frame.localMapPointIds_.size() < maxNumLocalMapPoints)
    {
        int keyframeId = covisibleKeyframeMap.begin()->first;
        auto keyframe = mapManager_->getKeyframe(keyframeId);
        
        // Find a valid keyframe
        while (!keyframe && keyframeId > 0)
        {
            keyframeId--;
            keyframe = mapManager_->getKeyframe(keyframeId);
        }

        if (keyframe)
        {
            frame.localMapPointIds_.insert(keyframe->localMapPointIds_.begin(), keyframe->localMapPointIds_.end());
            
            // go for another round if needed
            if (keyframe->keyframeId_ > 0 && frame.localMapPointIds_.size() < halfMaxPoints)
            {
                keyframe = mapManager_->getKeyframe(keyframe->keyframeId_);
                
                // Find another valid keyframe
                while (!keyframe && keyframeId > 0)
                {
                    keyframeId--;
                    keyframe = mapManager_->getKeyframe(keyframeId);
                }

                if (keyframe)
                {
                    frame.localMapPointIds_.insert(keyframe->localMapPointIds_.begin(), keyframe->localMapPointIds_.end());
                }
            }
        }
    }

    // Track local map
    std::map<int, int> mapPrevIdNewId = matchToMap(
        frame, 
        state_->mapMaxProjectionPxDistance_, 
        state_->mapMaxDescriptorDistance_, 
        frame.localMapPointIds_
    );

    // no matches
    if (mapPrevIdNewId.empty())
    {
        return false;
    }

    // Merge matches
    for (const auto &ids : mapPrevIdNewId)
    {
        mapManager_->mergeMapPoints(ids.first, ids.second);
    }

    return true;
}

std::map<int, int> Mapper::matchToMap(const Frame &frame, float maxProjectionError, float distRatio, 
                                     const std::unordered_set<int> &localMapPointIds)
{
    std::map<int, int> mapPrevIdNewId;

    // Leave if local map is empty
    if (localMapPointIds.empty())
    {
        return mapPrevIdNewId;
    }

    const float view_th = computeFOVThreshold(frame);

    // Define max distance from projection
    float maxPxDist = maxProjectionError;
    if (frame.numKeypoints3d_ < 30)
    {
        maxPxDist *= 2.0f;
    }

    // Use unordered_map for faster lookup
    std::unordered_map<int, std::vector<std::pair<int, float>>> keypointIdsMapPointIdsDist;
    keypointIdsMapPointIdsDist.reserve(localMapPointIds.size() / 2); // Estimate capacity

    // Go through all map point from the local map
    for (const int mapPointId : localMapPointIds)
    {
        if (frame.isObservingKeypoint(mapPointId))
        {
            continue;
        }

        auto mapPoint = mapManager_->getMapPoint(mapPointId);

        if (!mapPoint || !mapPoint->is3d_ || mapPoint->desc_.empty())
        {
            continue;
        }

        const Eigen::Vector3d &wpt = mapPoint->getPoint();

        //Project 3D map point into keyframe's image
        Eigen::Vector3d campt = frame.projWorldToCam(wpt);

        if (campt.z() < 0.1)
        {
            continue;
        }

        float view_angle = campt.z() / campt.norm();

        if (fabs(view_angle) < view_th)
        {
            continue;
        }

        cv::Point2f projPx = frame.projCamToImageDist(campt);

        if (!frame.isInImage(projPx))
        {
            continue;
        }

        // Get all the kps around the map point's projection
        const auto &nearKeyPoints = frame.getSurroundingKeypoints(projPx);

        // Find two best matches
        const float minDist = mapPoint->desc_.cols * distRatio * 8.0f; // * 8 to get bits size
        int bestId = -1;
        int secId = -1;
        float bestDist = minDist;
        float secDist = minDist;

        const std::set<int> &mapPointKeyframes = mapPoint->getObservedKeyframeIds();

        for (const auto &kp : nearKeyPoints)
        {
            if (kp.keypointId_ < 0)
            {
                continue;
            }

            float pxDist = cv::norm(projPx - kp.px_);

            if (pxDist > maxPxDist)
            {
                continue;
            }

            // Check that this kp and the map point are indeed candidates for matching
            auto kpMapPoint = mapManager_->getMapPoint(kp.keypointId_);

            if (!kpMapPoint)
            {
                mapManager_->removeMapPointObs(kp.keypointId_, frame.keyframeId_);
                continue;
            }

            if (kpMapPoint->desc_.empty())
            {
                continue;
            }

            // Check if they share any keyframes (not candidates if they do)
            bool isCandidate = true;
            for (const auto &keyframeId : kpMapPoint->getObservedKeyframeIds())
            {
                if (mapPointKeyframes.count(keyframeId) > 0)
                {
                    isCandidate = false;
                    break;
                }
            }

            if (!isCandidate)
            {
                continue;
            }

            float coProjectionPx = 0.0f;
            size_t numCoKeyPoints = 0;

            for (const auto &keyframeId : kpMapPoint->getObservedKeyframeIds())
            {
                auto coKeyframe = mapManager_->getKeyframe(keyframeId);
                if (coKeyframe)
                {
                    auto cokp = coKeyframe->getKeypointById(kp.keypointId_);
                    if (cokp.keypointId_ == kp.keypointId_)
                    {
                        coProjectionPx += cv::norm(cokp.px_ - coKeyframe->projWorldToImageDist(wpt));
                        numCoKeyPoints++;
                    }
                    else
                    {
                        mapManager_->removeMapPointObs(kp.keypointId_, keyframeId);
                    }
                }
                else
                {
                    mapManager_->removeMapPointObs(kp.keypointId_, keyframeId);
                }
            }

            if (numCoKeyPoints > 0 && coProjectionPx / numCoKeyPoints > maxPxDist)
            {
                continue;
            }

            float dist = mapPoint->computeMinDescDist(*kpMapPoint);

            if (dist <= bestDist)
            {
                secDist = bestDist;
                secId = bestId;
                bestDist = dist;
                bestId = kp.keypointId_;
            }
            else if (dist <= secDist)
            {
                secDist = dist;
                secId = kp.keypointId_;
            }
        }

        // Ratio test to ensure match uniqueness
        if (bestId != -1 && secId != -1 && 0.9 * secDist < bestDist)
        {
            bestId = -1;
        }

        if (bestId < 0)
        {
            continue;
        }

        // Use emplace_back instead of push_back
        auto it = keypointIdsMapPointIdsDist.find(bestId);
        if (it == keypointIdsMapPointIdsDist.end())
        {
            keypointIdsMapPointIdsDist.emplace(bestId, std::vector<std::pair<int, float>>{std::make_pair(mapPointId, bestDist)});
        }
        else
        {
            it->second.emplace_back(mapPointId, bestDist);
        }
    }

    // Find best map point for each keypoint
    for (const auto &keypointIdMapPointDist : keypointIdsMapPointIdsDist)
    {
        int keypointId = keypointIdMapPointDist.first;
        int bestMapPointId = -1;
        float bestDist = std::numeric_limits<float>::max();

        for (const auto &mapPointDist : keypointIdMapPointDist.second)
        {
            if (mapPointDist.second < bestDist)
            {
                bestDist = mapPointDist.second;
                bestMapPointId = mapPointDist.first;
            }
        }

        if (bestMapPointId >= 0)
        {
            mapPrevIdNewId.emplace(keypointId, bestMapPointId);
        }
    }

    return mapPrevIdNewId;
}