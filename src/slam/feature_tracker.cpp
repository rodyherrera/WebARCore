#include "feature_tracker.hpp"
#include <opencv2/video/tracking.hpp>
#include <iostream>

void FeatureTracker::fbKltTracking(const std::vector<cv::Mat> &prevPyramid, const std::vector<cv::Mat> &currPyramid,
                                   int winSize, int numPyramidLevels, float errorValue, float maxFbkltDistance,
                                   std::vector<cv::Point2f> &points,
                                   std::vector<cv::Point2f> &priorKeypoints, std::vector<bool> &keypointStatus) const{
    assert(prevPyramid.size() == currPyramid.size());
    if(points.empty()){
        return;
    }

    const int availableLevels = (int)prevPyramid.size() - 1;
    numPyramidLevels = std::min(numPyramidLevels, availableLevels);

    cv::Size kltWinSize(winSize, winSize);
    const size_t numKeypoints = points.size();

    keypointStatus.clear();
    keypointStatus.resize(numKeypoints, false);

    std::vector<uchar> status(numKeypoints, 0);
    std::vector<float> errors(numKeypoints, 0.0f);
    std::vector<int> keypointIndex; // Para hacer el tracking inverso solo de los buenos
    keypointIndex.reserve(numKeypoints);

    // Tracking Forward
    cv::calcOpticalFlowPyrLK(prevPyramid, currPyramid, points, priorKeypoints, status, errors,
                             kltWinSize, numPyramidLevels, kltConvCriteria_,
                             cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS);

    std::vector<cv::Point2f> newKeypoints;
    std::vector<cv::Point2f> backKeypoints;
    newKeypoints.reserve(numKeypoints);
    backKeypoints.reserve(numKeypoints);

    // Clasifica keypoints válidos y prepara para backward tracking
    for(size_t i = 0; i < numKeypoints; i++){
        if(!status[i]) continue;
        if(errors[i] > errorValue) continue;
        if(!inBorder(priorKeypoints[i], currPyramid[0])) continue;

        newKeypoints.push_back(priorKeypoints[i]);
        backKeypoints.push_back(points[i]);
        keypointStatus[i] = true;
        keypointIndex.push_back(i);
    }

    if(newKeypoints.empty()){
        return;
    }

    // Tracking Backward
    status.assign(newKeypoints.size(), 0);
    errors.assign(newKeypoints.size(), 0.0f);

    cv::calcOpticalFlowPyrLK(currPyramid, prevPyramid, newKeypoints, backKeypoints, status, errors,
                             kltWinSize, 0, kltConvCriteria_,
                             cv::OPTFLOW_USE_INITIAL_FLOW + cv::OPTFLOW_LK_GET_MIN_EIGENVALS);

    for(size_t i = 0; i < newKeypoints.size(); i++){
        int idx = keypointIndex[i];
        if(!status[i]){
            keypointStatus[idx] = false;
            continue;
        }
        if(cv::norm(points[idx] - backKeypoints[i]) > maxFbkltDistance){
            keypointStatus[idx] = false;
            continue;
        }
    }
    // if(state_->debug_){
    //     std::cout << "- [FeatureTracker]: fbKltTracking - " << std::count(keypointStatus.begin(), keypointStatus.end(), true) << " / " << numKeypoints << " tracked\n";
    // }
}

bool FeatureTracker::inBorder(const cv::Point2f &point, const cv::Mat &image) const{
    const float BORDER_SIZE = 1.0f;
    return BORDER_SIZE <= point.x && point.x < image.cols - BORDER_SIZE &&
           BORDER_SIZE <= point.y && point.y < image.rows - BORDER_SIZE;
}