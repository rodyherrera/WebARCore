#pragma once

#include <unordered_map>
#include <vector>
#include "state.hpp"
#include "frame.hpp"
#include "map_point.hpp"
#include "feature_extractor.hpp"
#include "feature_tracker.hpp"

class MapManager {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MapManager() = default;
    MapManager(std::shared_ptr<State> state, std::shared_ptr<Frame> frame, std::shared_ptr<FeatureExtractor> featureExtractor);

    // Core map management functions
    void createKeyframe(const cv::Mat &image, const cv::Mat &imageRaw);
    void prepareFrame();
    void addKeyframe();
    void updateFrameCovisibility(Frame &frame);
    
    // Map point management
    void addMapPoint(const cv::Scalar &color = cv::Scalar(200));
    void addMapPoint(const cv::Mat &desc, const cv::Scalar &color = cv::Scalar(200));
    void updateMapPoint(const int mapPointId, const Eigen::Vector3d &wpt, const double keyframeAnchorInvDepth = -1.);
    bool setMapPointObs(const int mapPointId);
    void mergeMapPoints(const int prvMapPointId, const int newMapPointId);
    void removeMapPoint(const int mapPointId);
    void removeMapPointObs(const int mapPointId, const int keyframeId);
    void removeObsFromCurrFrameById(const int mapPointId);

    // Keyframe management
    void removeKeyframe(const int keyframeId);
    
    // Feature operations
    void addKeypointsToFrame(const cv::Mat &image, const std::vector<cv::Point2f> &points, 
                            const std::vector<cv::Mat> &descriptors, Frame &frame);
    void extractKeypoints(const cv::Mat &image, const cv::Mat &imageRaw);
    void describeKeypoints(const cv::Mat &image, const std::vector<Keypoint> &keypoints, 
                          const std::vector<cv::Point2f> &points);

    // Getters
    std::shared_ptr<Frame> getKeyframe(const int KeyframeId) const;
    std::shared_ptr<MapPoint> getMapPoint(const int mapPointId) const;
    std::vector<Eigen::Vector3d> getCurrentFrameMapPoints() const;

    void reset();

    // Public properties
    int numMapPointIds_ = 0;
    int numKeyframeIds_ = 0;
    int numMapPoints_ = 0;
    int numKeyframes_ = 0;

    std::shared_ptr<State> state_;
    std::shared_ptr<Frame> currFrame_;
    std::shared_ptr<FeatureExtractor> featureExtractor_;

    std::unordered_map<int, std::shared_ptr<Frame>> mapKeyframes_;
    std::unordered_map<int, std::shared_ptr<MapPoint>> mapMapPoints_;
    std::vector<Point3D> pointCloud_;

private:
    // Helper to update point cloud entry
    inline void updatePointCloudEntry(int id, const Eigen::Vector3d& position, const cv::Scalar& color);
    inline void updatePointCloudEntry(int id, const Eigen::Vector3d& position, bool isObserved);
};