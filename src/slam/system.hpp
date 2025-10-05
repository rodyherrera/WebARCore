
#pragma once

#include <memory>
#include <vector>
#include <queue>
#include <chrono>
#include <Eigen/Core>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <sophus/se3.hpp>
#include "../core/frame.hpp"
#include "../core/state.hpp"
#include "../core/utils.hpp"
#include "../vision/camera_calibration.hpp"
#include "../vision/feature_extractor.hpp"
#include "../vision/feature_tracker.hpp"
#include "../core/one_euro_filter.hpp"
#include "mapper.hpp"
#include "map_manager.hpp"
#include "visual_frontend.hpp"

class System
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    System();

    ~System();

    void configure(int imageWidth, int imageHeight, double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2);

    void reset();
    
    void enablePoseSmoothing(bool enable, double minCutoffPos = 1.0, double minCutoffRot = 1.5, double beta = 0.007);
    bool isPoseSmoothingEnabled() const { return poseSmoothingEnabled_; }

    int findCameraPoseWithIMU(int imageRGBADataPtr, int imuDataPtr, int posePtr);
    cv::Mat fastPlaneDetection(const std::vector<Eigen::Vector3d>& points, const Sophus::SE3d& Twc, int numIterations);

    int findCameraPose(int imageRGBADataPtr, int posePtr);

    int findPlane(int locationPtr, int numIterations);

    int getFramePoints(int pointsPtr);
    
    double getLastFrameTime();
    int getCurrentFPS();
    void enablePerformanceMonitoring(bool enable);

private:
    int processCameraPose(cv::Mat &image, uint64_t timestamp);
    uint64_t getTimestamp();

    std::shared_ptr<State> state_;
    std::shared_ptr<Frame> currFrame_;
    std::shared_ptr<CameraCalibration> cameraCalibration_;
    std::shared_ptr<MapManager> mapManager_;
    std::shared_ptr<Mapper> mapper_;
    std::unique_ptr<VisualFrontend> visualFrontend_;
    std::shared_ptr<FeatureExtractor> featureExtractor_;
    std::shared_ptr<FeatureTracker> featureTracker_;

    Eigen::Vector3d currTranslation_;
    Eigen::Vector3d prevTranslation_;
    
    bool performanceMonitoringEnabled_ = false;
    std::chrono::high_resolution_clock::time_point lastFrameTime_;
    std::chrono::high_resolution_clock::time_point currentFrameTime_;
    double lastFrameDuration_ = 0.0;
    int frameCount_ = 0;
    std::chrono::high_resolution_clock::time_point fpsStartTime_;
    
    // One Euro Filter for pose smoothing
    bool poseSmoothingEnabled_ = false;
    std::unique_ptr<OneEuroFilterSE3> poseFilter_;
};
