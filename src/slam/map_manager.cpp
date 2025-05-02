#include <opencv2/highgui.hpp>
#include "multi_view_geometry.hpp"
#include "map_manager.hpp"

constexpr size_t INITIAL_POINT_CLOUD_CAPACITY = 10000; // Reduced from 1e5
constexpr unsigned char OBSERVED_POINT_COLOR = 255;
constexpr unsigned char DEFAULT_POINT_COLOR = 200;

MapManager::MapManager(std::shared_ptr<State> state, std::shared_ptr<Frame> frame, std::shared_ptr<FeatureExtractor> featureExtractor)
    : numMapPointIds_(0), 
      numKeyframeIds_(0), 
      numMapPoints_(0), 
      numKeyframes_(0), 
      state_(std::move(state)), 
      currFrame_(std::move(frame)), 
      featureExtractor_(std::move(featureExtractor))
{
    pointCloud_.clear();
    pointCloud_.reserve(INITIAL_POINT_CLOUD_CAPACITY);
}

void MapManager::createKeyframe(const cv::Mat &image, const cv::Mat &imageRaw)
{
    prepareFrame();
    extractKeypoints(image, imageRaw);
    addKeyframe();
}

void MapManager::prepareFrame()
{
    currFrame_->keyframeId_ = numKeyframeIds_;

    // Filter if too many keypoints
    if (currFrame_->numKeypoints_ > state_->frameMaxNumKeypoints_) {
        for (const auto &keypointIds : currFrame_->gridKeypointsIds_) {
            if (keypointIds.size() <= 2) continue;
            
            int mapPointIdToRemove = -1;
            size_t minNumObs = std::numeric_limits<size_t>::max();

            for (const auto &lmid : keypointIds) {
                auto it = mapMapPoints_.find(lmid);

                if (it != mapMapPoints_.end()) {
                    size_t numObs = it->second->getObservedKeyframeIds().size();
                    if (numObs < minNumObs) {
                        mapPointIdToRemove = lmid;
                        minNumObs = numObs;
                    }
                } else {
                    removeObsFromCurrFrameById(lmid);
                    mapPointIdToRemove = -1; // We already removed one point, no need to remove another
                    break;
                }
            }
            
            if (mapPointIdToRemove >= 0) {
                removeObsFromCurrFrameById(mapPointIdToRemove);
            }
        }
    }

    // Update observed keyframe IDs for map points
    const auto& keypoints = currFrame_->getKeypoints();
    for (const auto &kp : keypoints) {
        auto it = mapMapPoints_.find(kp.keypointId_);
        if (it == mapMapPoints_.end()) {
            removeObsFromCurrFrameById(kp.keypointId_);
            continue;
        }
        
        it->second->addObservedKeyframeId(numKeyframeIds_);
    }
}

void MapManager::updateFrameCovisibility(Frame &frame)
{
    std::map<int, int> mapCovisibleKeyframes;
    std::unordered_set<int> localMapIds;

    // Build covisibility map and collect local map IDs
    const auto& keypoints = frame.getKeypoints();
    for (const auto &kp : keypoints) {
        auto it = mapMapPoints_.find(kp.keypointId_);
        if (it == mapMapPoints_.end()) {
            removeMapPointObs(kp.keypointId_, frame.keyframeId_);
            removeObsFromCurrFrameById(kp.keypointId_);
            continue;
        }

        // Update covisibility for keyframes observing this point
        for (const int kfid : it->second->getObservedKeyframeIds()) {
            if (kfid != frame.keyframeId_) {
                auto& count = mapCovisibleKeyframes[kfid];
                count += 1; // Using direct insert/increment
            }
        }
    }

    // Update covisibility for keyframes
    std::vector<int> badKeyframeIds;
    for (const auto &[keyframeId, covScore] : mapCovisibleKeyframes) {
        auto it = mapKeyframes_.find(keyframeId);
        if (it != mapKeyframes_.end()) {
            // Update covisibility score
            it->second->covisibleKeyframeIds_[frame.keyframeId_] = covScore;
            
            // Collect unobserved local map points
            for (const auto &kp : it->second->getKeypoints3d()) {
                if (!frame.isObservingKeypoint(kp.keypointId_)) {
                    localMapIds.insert(kp.keypointId_);
                }
            }
        } else {
            badKeyframeIds.push_back(keyframeId);
        }
    }

    // Remove bad keyframe IDs
    for (int id : badKeyframeIds) {
        mapCovisibleKeyframes.erase(id);
    }

    // Update covisibility and local map info
    frame.covisibleKeyframeIds_.swap(mapCovisibleKeyframes);
    
    if (localMapIds.size() > 0.5 * frame.localMapPointIds_.size()) {
        frame.localMapPointIds_.swap(localMapIds);
    } else {
        frame.localMapPointIds_.insert(localMapIds.begin(), localMapIds.end());
    }
}

void MapManager::addKeypointsToFrame(const cv::Mat &image, 
                                    const std::vector<cv::Point2f> &points,
                                    const std::vector<cv::Mat> &descriptors, 
                                    Frame &frame)
{
    const size_t numPoints = points.size();
    for (size_t i = 0; i < numPoints; i++) {
        const auto& point = points[i];
        const cv::Scalar pixel = image.at<uchar>(point.y, point.x);
        
        if (i < descriptors.size() && !descriptors[i].empty()) {
            frame.addKeypoint(point, numMapPointIds_, descriptors[i]);
            addMapPoint(descriptors[i], pixel);
        } else {
            frame.addKeypoint(point, numMapPointIds_);
            addMapPoint(pixel);
        }
    }
}

void MapManager::extractKeypoints(const cv::Mat &image, const cv::Mat &imageRaw)
{
    // Describe existing keypoints
    const auto& keypoints = currFrame_->getKeypoints();
    
    if (!keypoints.empty()) {
        std::vector<cv::Point2f> points;
        points.reserve(keypoints.size());
        
        for (const auto &kp : keypoints) {
            points.push_back(kp.px_);
        }
        
        describeKeypoints(imageRaw, keypoints, points);
    }

    // Detect new keypoints if needed
    int numToDetect = state_->frameMaxNumKeypoints_ - currFrame_->numOccupiedCells_;
    if (numToDetect > 0) {
        std::vector<cv::Point2f> existingPoints;
        existingPoints.reserve(keypoints.size());
        
        for (const auto &kp : keypoints) {
            existingPoints.push_back(kp.px_);
        }
        
        std::vector<cv::Point2f> newPoints = featureExtractor_->detectFeaturePoints(
            image, 
            state_->frameMaxCellSize_, 
            existingPoints, 
            currFrame_->cameraCalibration_->roi_rect_
        );

        if (!newPoints.empty()) {
            std::vector<cv::Mat> descriptors = featureExtractor_->describeFeaturePoints(imageRaw, newPoints);
            addKeypointsToFrame(image, newPoints, descriptors, *currFrame_);
        }
    }
}

void MapManager::describeKeypoints(const cv::Mat &image, 
                                  const std::vector<Keypoint> &keypoints, 
                                  const std::vector<cv::Point2f> &points)
{
    if (points.empty()) return;
    
    std::vector<cv::Mat> descriptors = featureExtractor_->describeFeaturePoints(image, points);
    
    if (descriptors.size() != keypoints.size()) return;
    
    for (size_t i = 0; i < keypoints.size(); i++) {
        if (!descriptors[i].empty()) {
            const int keypointId = keypoints[i].keypointId_;
            currFrame_->updateKeypointDesc(keypointId, descriptors[i]);
            
            auto it = mapMapPoints_.find(keypointId);
            if (it != mapMapPoints_.end()) {
                it->second->addDesc(currFrame_->keyframeId_, descriptors[i]);
            }
        }
    }
}

void MapManager::addKeyframe()
{
    // Create a copy of current frame shared_ptr for creating an independent keyframe
    std::shared_ptr<Frame> keyframe = std::allocate_shared<Frame>(
        Eigen::aligned_allocator<Frame>(), 
        *currFrame_
    );

    // Add keyframe to the map and update counters
    mapKeyframes_.emplace(numKeyframeIds_, std::move(keyframe));
    numKeyframes_++;
    numKeyframeIds_++;
}

void MapManager::addMapPoint(const cv::Scalar &color)
{
    std::shared_ptr<MapPoint> mapPoint = std::allocate_shared<MapPoint>(
        Eigen::aligned_allocator<MapPoint>(),
        numMapPointIds_,
        numKeyframeIds_,
        color
    );

    mapMapPoints_.emplace(numMapPointIds_, std::move(mapPoint));
    
    Point3D point;
    point.r = color[0];
    point.g = color[0];
    point.b = color[0];
    point.x = 0.0;
    point.y = 0.0;
    point.z = 0.0;
    
    pointCloud_.push_back(point);
    
    numMapPointIds_++;
    numMapPoints_++;
}

void MapManager::addMapPoint(const cv::Mat &desc, const cv::Scalar &color)
{
    std::shared_ptr<MapPoint> mapPoint = std::allocate_shared<MapPoint>(
        Eigen::aligned_allocator<MapPoint>(),
        numMapPointIds_,
        numKeyframeIds_,
        desc,
        color
    );

    mapMapPoints_.emplace(numMapPointIds_, std::move(mapPoint));
    
    Point3D point;
    point.r = color[0];
    point.g = color[0];
    point.b = color[0];
    point.x = 0.0;
    point.y = 0.0;
    point.z = 0.0;
    
    pointCloud_.push_back(point);
    
    numMapPointIds_++;
    numMapPoints_++;
}

std::shared_ptr<Frame> MapManager::getKeyframe(const int keyframeId) const
{
    auto it = mapKeyframes_.find(keyframeId);
    return (it != mapKeyframes_.end()) ? it->second : nullptr;
}

std::shared_ptr<MapPoint> MapManager::getMapPoint(const int mapPointId) const
{
    auto it = mapMapPoints_.find(mapPointId);
    return (it != mapMapPoints_.end()) ? it->second : nullptr;
}

std::vector<Eigen::Vector3d> MapManager::getCurrentFrameMapPoints() const
{
    std::vector<Eigen::Vector3d> mapPoints;
    mapPoints.reserve(numMapPoints_ / 2);  // Reserve approximate capacity

    for (const auto& [id, mapPoint] : mapMapPoints_) {
        if (mapPoint->isObserved_ && mapPoint->is3d_) {
            mapPoints.push_back(mapPoint->point3d_);
        }
    }

    return mapPoints;
}

inline void MapManager::updatePointCloudEntry(int id, const Eigen::Vector3d& position, const cv::Scalar& color)
{
    if (id >= pointCloud_.size()) return;
    
    Point3D& point = pointCloud_[id];
    point.r = color[0];
    point.g = color[0];
    point.b = color[0];
    point.x = position.x();
    point.y = position.y();
    point.z = position.z();
}

inline void MapManager::updatePointCloudEntry(int id, const Eigen::Vector3d& position, bool isObserved)
{
    if (id >= pointCloud_.size()) return;
    
    Point3D& point = pointCloud_[id];
    point.r = isObserved ? OBSERVED_POINT_COLOR : mapMapPoints_[id]->color_[0];
    point.g = isObserved ? 0 : mapMapPoints_[id]->color_[0];
    point.b = isObserved ? 0 : mapMapPoints_[id]->color_[0];
    point.x = position.x();
    point.y = position.y();
    point.z = position.z();
}

void MapManager::updateMapPoint(const int mapPointId, const Eigen::Vector3d &wpt, const double keyframeAnchorInvDepth)
{
    auto it = mapMapPoints_.find(mapPointId);
    if (it == mapMapPoints_.end() || !it->second) return;

    // Handle 2D to 3D conversion
    if (!it->second->is3d_) {
        for (const int keyframeId : it->second->getObservedKeyframeIds()) {
            auto kfIt = mapKeyframes_.find(keyframeId);
            if (kfIt != mapKeyframes_.end()) {
                kfIt->second->turnKeypoint3d(mapPointId);
            } else {
                it->second->removeObservedKeyframeId(keyframeId);
            }
        }

        if (it->second->isObserved_) {
            currFrame_->turnKeypoint3d(mapPointId);
        }
    }

    // Update map point position
    if (keyframeAnchorInvDepth >= 0.) {
        it->second->setPoint(wpt, keyframeAnchorInvDepth);
    } else {
        it->second->setPoint(wpt);
    }

    // Update point cloud
    updatePointCloudEntry(mapPointId, wpt, it->second->isObserved_);
}

void MapManager::mergeMapPoints(const int prvMapPointId, const int newMapPointId)
{
    auto prvIt = mapMapPoints_.find(prvMapPointId);
    auto newIt = mapMapPoints_.find(newMapPointId);
    
    if (prvIt == mapMapPoints_.end() || newIt == mapMapPoints_.end() || !newIt->second->is3d_) {
        return;
    }

    // Get data from previous map point
    const std::set<int>& prevKfIds = prvIt->second->getObservedKeyframeIds();
    const std::set<int>& nextKfIds = newIt->second->getObservedKeyframeIds();
    const auto& mapPrevKfDesc = prvIt->second->mapKeyframeDescriptors_;

    // Update keyframes and covisibility
    for (const int pkfid : prevKfIds) {
        auto pkfIt = mapKeyframes_.find(pkfid);
        if (pkfIt != mapKeyframes_.end()) {
            if (pkfIt->second->updateKeypointId(prvMapPointId, newMapPointId, newIt->second->is3d_)) {
                newIt->second->addObservedKeyframeId(pkfid);
                
                // Update covisibility
                for (const int nkfid : nextKfIds) {
                    auto nkfIt = mapKeyframes_.find(nkfid);
                    if (nkfIt != mapKeyframes_.end()) {
                        pkfIt->second->addCovisibleKeyframe(nkfid);
                        nkfIt->second->addCovisibleKeyframe(pkfid);
                    }
                }
            }
        }
    }

    // Transfer descriptors
    for (const auto& [kfid, desc] : mapPrevKfDesc) {
        newIt->second->addDesc(kfid, desc);
    }

    // Update current frame if needed
    if (currFrame_->isObservingKeypoint(prvMapPointId)) {
        if (currFrame_->updateKeypointId(prvMapPointId, newMapPointId, newIt->second->is3d_)) {
            setMapPointObs(newMapPointId);
        }
    }

    // Update counters
    if (prvIt->second->is3d_) {
        numMapPoints_--;
    }

    // Remove old map point
    mapMapPoints_.erase(prvIt);
    
    // Clear point in point cloud
    Point3D point = {};
    pointCloud_[prvMapPointId] = point;
}

void MapManager::removeKeyframe(const int keyframeId)
{
    auto kfIt = mapKeyframes_.find(keyframeId);
    if (kfIt == mapKeyframes_.end()) return;

    // Remove from map points' observations
    for (const auto &kp : kfIt->second->getKeypoints()) {
        auto mpIt = mapMapPoints_.find(kp.keypointId_);
        if (mpIt != mapMapPoints_.end()) {
            mpIt->second->removeObservedKeyframeId(keyframeId);
        }
    }

    // Update covisibility graph
    for (const auto &[covisibleKfId, _] : kfIt->second->getCovisibleKeyframeMap()) {
        auto covisibleKfIt = mapKeyframes_.find(covisibleKfId);
        if (covisibleKfIt != mapKeyframes_.end()) {
            covisibleKfIt->second->removeCovisibleKeyframe(keyframeId);
        }
    }

    // Remove keyframe and update counter
    mapKeyframes_.erase(kfIt);
    numKeyframes_--;

    if (state_->debug_) {
        std::cout << "- [Map-Manager]: Remove keyframe #" << keyframeId << std::endl;
    }
}

void MapManager::removeMapPoint(const int mapPointId)
{
    auto mpIt = mapMapPoints_.find(mapPointId);
    if (mpIt == mapMapPoints_.end()) return;

    // Remove observations from keyframes
    for (const int kfId : mpIt->second->getObservedKeyframeIds()) {
        auto kfIt = mapKeyframes_.find(kfId);
        if (kfIt == mapKeyframes_.end()) continue;
        
        kfIt->second->removeKeypointById(mapPointId);
        
        // Update covisibility graph
        for (const int covisibleKfId : mpIt->second->getObservedKeyframeIds()) {
            if (covisibleKfId != kfId) {
                kfIt->second->decreaseCovisibleKeyframe(covisibleKfId);
            }
        }
    }

    // Remove from current frame if observed
    if (mpIt->second->isObserved_) {
        currFrame_->removeKeypointById(mapPointId);
    }

    // Update counters
    if (mpIt->second->is3d_) {
        numMapPoints_--;
    }

    // Remove map point
    mapMapPoints_.erase(mpIt);
    
    // Clear point in point cloud
    Point3D point = {};
    pointCloud_[mapPointId] = point;
}

void MapManager::removeMapPointObs(const int mapPointId, const int keyframeId)
{
    // Remove from keyframe
    auto kfIt = mapKeyframes_.find(keyframeId);
    if (kfIt != mapKeyframes_.end()) {
        kfIt->second->removeKeypointById(mapPointId);
    }
    
    // Remove keyframe from map point observations
    auto mpIt = mapMapPoints_.find(mapPointId);
    if (mpIt == mapMapPoints_.end()) return;
    
    mpIt->second->removeObservedKeyframeId(keyframeId);
    
    // Update covisibility graph
    if (kfIt != mapKeyframes_.end()) {
        for (const int covisibleKfId : mpIt->second->getObservedKeyframeIds()) {
            auto covisibleKfIt = mapKeyframes_.find(covisibleKfId);
            if (covisibleKfIt != mapKeyframes_.end()) {
                kfIt->second->decreaseCovisibleKeyframe(covisibleKfId);
                covisibleKfIt->second->decreaseCovisibleKeyframe(keyframeId);
            }
        }
    }
}

void MapManager::removeObsFromCurrFrameById(const int mapPointId)
{
    // Remove from current frame
    currFrame_->removeKeypointById(mapPointId);

    // Update map point status
    auto it = mapMapPoints_.find(mapPointId);
    if (it == mapMapPoints_.end()) {
        // Point doesn't exist, just clear its entry in point cloud
        Point3D point = {};
        pointCloud_[mapPointId] = point;
        return;
    }

    // Mark as not observed
    it->second->isObserved_ = false;
    
    // Update color in point cloud (keep position)
    Point3D& point = pointCloud_[mapPointId];
    point.r = it->second->color_[0];
    point.g = it->second->color_[0];
    point.b = it->second->color_[0];
}

bool MapManager::setMapPointObs(const int mapPointId)
{
    auto it = mapMapPoints_.find(mapPointId);
    if (it == mapMapPoints_.end()) {
        // Point doesn't exist, clear its entry in point cloud
        Point3D point = {};
        pointCloud_[mapPointId] = point;
        return false;
    }

    // Mark as observed
    it->second->isObserved_ = true;
    
    // Update color in point cloud (keep position)
    Point3D& point = pointCloud_[mapPointId];
    point.r = OBSERVED_POINT_COLOR;
    point.g = 0;
    point.b = 0;
    
    return true;
}

void MapManager::reset()
{
    numMapPointIds_ = 0;
    numKeyframeIds_ = 0;
    numMapPoints_ = 0;
    numKeyframes_ = 0;

    mapKeyframes_.clear();
    mapMapPoints_.clear();
    pointCloud_.clear();
}