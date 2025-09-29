#include "system.hpp"
#include <iostream>
#include <numeric>
#include <random>

System::System() = default;
System::~System() = default;

void System::configure(int imageWidth, int imageHeight, double fx, double fy, double cx, double cy, double k1, double k2, double p1, double p2){
    state_ = std::make_shared<State>(imageWidth, imageHeight, 60); // Increased for max performance
    state_->debug_ = false;
    state_->claheEnabled_ = false; // Disabled for max performance
    state_->mapKeyframeFilteringRatio_ = 0.85; // Reduced for max performance
    state_->p3pEnabled_ = true;
    
    // Optimize for maximum performance
    state_->kltPyramidLevels_ = 2; // Reduced pyramid levels
    state_->multiViewRansacNumIterations_ = 50; // Reduced RANSAC iterations
    state_->trackerMaxIterations_ = 20; // Reduced tracker iterations
    state_->kltError_ = 25.0; // Slightly relaxed error threshold
    state_->minAvgRotationParallax_ = 30.0; // Reduced parallax threshold

    cameraCalibration_ = std::make_shared<CameraCalibration>(fx, fy, cx, cy, k1, k2, p1, p2, imageWidth, imageHeight, 20);
    currFrame_ = std::make_shared<Frame>(cameraCalibration_, state_->frameMaxCellSize_);

    featureExtractor_ = std::make_shared<FeatureExtractor>(state_->extractorMaxQuality_);
    featureTracker_ = std::make_shared<FeatureTracker>(state_->trackerMaxIterations_, state_->trackerMaxPxPrecision_);

    mapManager_ = std::make_shared<MapManager>(state_, currFrame_, featureExtractor_);
    mapper_ = std::make_shared<Mapper>(state_, mapManager_, currFrame_);
    visualFrontend_ = std::make_unique<VisualFrontend>(state_, currFrame_, mapManager_, mapper_, featureTracker_);
}

void System::reset(){
    if(state_->debug_){
        std::cout << "- [System]: Reset\n";
    }
    currFrame_->reset();
    visualFrontend_->reset();
    mapManager_->reset();
    state_->reset();
    prevTranslation_.setZero();
}

int System::findCameraPoseWithIMU(int imageRGBADataPtr, int imuDataPtr, int posePtr){
    auto* imageData = reinterpret_cast<uint8_t*>(imageRGBADataPtr);
    auto* imuData = reinterpret_cast<double*>(imuDataPtr);
    auto* poseData = reinterpret_cast<float*>(posePtr);

    cv::Mat imageRGBA(state_->imgHeight_, state_->imgWidth_, CV_8UC4, imageData);
    cv::Mat gray;
    cv::cvtColor(imageRGBA, gray, cv::COLOR_RGBA2GRAY);

    Eigen::Quaterniond qimu(imuData[0], -imuData[1], imuData[2], imuData[3]);
    Eigen::Matrix3d Rwc = qimu.toRotationMatrix().inverse();
    Sophus::SE3d Twc(Rwc, Eigen::Vector3d::Zero());

    int motionSampleSize = 7;
    int status = processCameraPose(gray, getTimestamp());

    if(status == 1){
        Eigen::Vector3d t = currFrame_->getTwc().translation();
        currTranslation_ += t - prevTranslation_;
        prevTranslation_ = t;
    }else{
        prevTranslation_.setZero();
    }

    Twc.translation() = currTranslation_;
    Utils::toPoseArray(Twc, poseData);
    return 1;
}

int System::findCameraPose(int imageRGBADataPtr, int posePtr){
    auto* imageData = reinterpret_cast<uint8_t*>(imageRGBADataPtr);
    auto* poseData = reinterpret_cast<float*>(posePtr);

    cv::Mat imageRGBA(state_->imgHeight_, state_->imgWidth_, CV_8UC4, imageData);
    cv::Mat gray;
    cv::cvtColor(imageRGBA, gray, cv::COLOR_RGBA2GRAY);

    int status = processCameraPose(gray, getTimestamp());
    Utils::toPoseArray(currFrame_->getTwc(), poseData);
    return status;
}

int System::getFramePoints(int pointsPtr){
    auto* data = reinterpret_cast<int*>(pointsPtr);
    const auto& keypoints = currFrame_->getKeypoints2d();
    int n = std::min((int) keypoints.size() * 2, 4096);
    for(int i = 0, j = 0; i < n / 2; i++){
        const auto &p = keypoints[i].unpx_;
        data[j++] = static_cast<int>(p.x);
        data[j++] = static_cast<int>(p.y);
    }
    return (int) keypoints.size();
}

int System::processCameraPose(cv::Mat& image, uint64_t timestamp){
    if(performanceMonitoringEnabled_){
        currentFrameTime_ = std::chrono::high_resolution_clock::now();
        if(frameCount_ > 0){
            lastFrameDuration_ = std::chrono::duration<double, std::milli>(currentFrameTime_ - lastFrameTime_).count();
        }
        lastFrameTime_ = currentFrameTime_;
        frameCount_++;
    }
    
    currFrame_->id_++;
    currFrame_->timestamp_ = timestamp;

    visualFrontend_->track(image, timestamp);
    if(state_->slamResetRequested_){
        reset();
        return 2;
    }

    if(!state_->slamReadyForInit_) return 3;
    return 1;
}

uint64_t System::getTimestamp() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch()).count();
}


int System::findPlane(int locationPtr, int numIterations){
    const auto& pts = mapManager_->getCurrentFrameMapPoints();
    if(pts.size() < 32) return 0;
    cv::Mat plane = fastPlaneDetection(pts, currFrame_->getTwc(), numIterations);
    if(plane.empty()) return 0;
    auto* poseData = reinterpret_cast<float*>(locationPtr);
    Utils::toPoseArray(plane, poseData);
    return 1;
}

cv::Mat System::fastPlaneDetection(const std::vector<Eigen::Vector3d>& points, const Sophus::SE3d& Twc, int numIterations){
    const int N = (int) points.size();
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);

    float bestError = 1e10;
    Eigen::Vector4f bestPlane;
    std::vector<float> distances(N);

    std::mt19937 rng{std::random_device{}()};

    for(int i = 0; i < numIterations; i++){
        std::shuffle(indices.begin(), indices.end(), rng);
        Eigen::MatrixXf A(3, 4);
        A.block<3, 3>(0, 0) << points[indices[0]].cast<float>().transpose(),
                        points[indices[1]].cast<float>().transpose(),
                        points[indices[2]].cast<float>().transpose();
        A.block<3, 1>(0, 3).setOnes();
        Eigen::JacobiSVD<Eigen::MatrixXf> svd(A, Eigen::ComputeFullV);
        Eigen::Vector4f plane = svd.matrixV().col(3);
        Eigen::Vector3f normal = plane.head<3>();
        float norm = normal.norm();
        if(norm == 0) continue;
        plane /= norm;
        for(int j = 0; j < N; ++j){
            distances[j] = std::abs(plane.head<3>().dot(points[j].template cast<float>()) + plane[3]);
        }
        int medianIndex = std::max(20, N / 5);
        std::nth_element(distances.begin(), distances.begin() + medianIndex, distances.end());
        float median = distances[medianIndex];
        if(median < bestError){
            bestError = median;
            bestPlane = plane;
        }
    }
    
    if(bestError > 1.0f){
        return cv::Mat();
    }

    Eigen::Vector3f t = Eigen::Vector3f::Zero();
    for(const auto &p : points){
        t += p.cast<float>();
    }
    t /= points.size();

    Eigen::Matrix4f pose = Eigen::Matrix4f::Identity();
    pose.block<3, 1>(0, 3) = t;
    //pose.block<3, 3>(0, 0) = Eigen::Quaternionf().setFromTwoVectors(Eigen::Vector3f::UnitZ(), bestPlane.head<3>()).toRotationMatrix();
    Eigen::Vector3f z = bestPlane.head<3>();
    // Invalid plane
    if(z.norm() < 1e-5) return cv::Mat();

    z.normalize();
    Eigen::Quaternionf q;
    if(z.isApprox(Eigen::Vector3f::UnitZ())){
        q.setIdentity();
    }else if(z.isApprox(-Eigen::Vector3f::UnitZ())){
        q = Eigen::Quaternionf(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitX()));
    }else{
        q = Eigen::Quaternionf().setFromTwoVectors(Eigen::Vector3f::UnitZ(), z);
    }
    pose.block<3,3>(0,0) = q.toRotationMatrix();

    cv::Mat result;
    cv::eigen2cv(pose, result);
    return result;
}

double System::getLastFrameTime(){
    return lastFrameDuration_;
}

int System::getCurrentFPS(){
    if(!performanceMonitoringEnabled_ || frameCount_ == 0){
        return 0;
    }
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - fpsStartTime_);
    if(duration.count() > 0){
        return static_cast<int>(frameCount_ * 1000.0 / duration.count());
    }
    return 0;
}

void System::enablePerformanceMonitoring(bool enable){
    performanceMonitoringEnabled_ = enable;
    if(enable){
        fpsStartTime_ = std::chrono::high_resolution_clock::now();
        frameCount_ = 0;
    }
}