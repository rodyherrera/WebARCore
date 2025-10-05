#pragma once

#include <opencv2/opencv.hpp>
#include <vector>

class FeatureExtractor{
public:
    explicit FeatureExtractor(double maxQuality = 0.01);

    std::vector<cv::Point2f> detectFeaturePoints(
        const cv::Mat &image,
        int cellSize,
        const std::vector<cv::Point2f> &currKeypoints,
        const cv::Rect &roi);
    
    std::vector<cv::Mat> describeFeaturePoints(
        const cv::Mat &image,
        const std::vector<cv::Point2f> &points) const;
    
private:
    double maxQuality_;
    cv::Ptr<cv::DescriptorExtractor> descriptor_;
};