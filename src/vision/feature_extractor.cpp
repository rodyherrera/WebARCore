#include "feature_extractor.hpp"
#include <algorithm>
#include <iterator>
#include <unordered_map>
#include <thread>

cv::Ptr<cv::DescriptorExtractor> descriptor_;

FeatureExtractor::FeatureExtractor(double maxQuality) 
    : maxQuality_(maxQuality),
    descriptor_(cv::ORB::create(500, 1.0, 0)){}

std::vector<cv::Point2f> FeatureExtractor::detectFeaturePoints(
    const cv::Mat &image, 
    const int cellSize, 
    const std::vector<cv::Point2f> &currKeypoints, 
    const cv::Rect &roi)
{
    if(image.empty()){
        return {};
    }

    const size_t numCols = image.cols;
    const size_t numRows = image.rows;
    const size_t cellSizeHalf = cellSize / 4;
    const size_t numCellsW = numCols / cellSize;
    const size_t numCellsH = numRows / cellSize;
    const size_t numCells = numCellsH * numCellsW;

    std::vector<cv::Point2f> detectedPx;
    detectedPx.reserve(numCells);

    std::vector<bool> occupiedCells(numCells, false);
    cv::Mat mask = cv::Mat::ones(image.rows, image.cols, CV_32F);

    size_t numOccupied = 0;
    for(const auto &px : currKeypoints){
        size_t cellIdx = (size_t)(px.y / cellSize) * numCellsW + (size_t)(px.x / cellSize);
        if(cellIdx < numCells){
            occupiedCells[cellIdx] = true;
            numOccupied++;
            cv::circle(mask, px, cellSizeHalf, cv::Scalar(0.), -1);
        }
    }

    std::vector<std::vector<cv::Point2f>> primaryDetections(numCells);
    std::vector<std::vector<cv::Point2f>> secondaryDetections(numCells);

    // Pre-allocate matrices for better performance
    cv::Mat filteredImage, hMap;
    
    // Use more efficient parallel processing with better load balancing
    const int numThreads = std::min(static_cast<int>(std::thread::hardware_concurrency()), 8);
    const int cellsPerThread = std::max(1, static_cast<int>(numCells) / numThreads);
    
        cv::parallel_for_(cv::Range(0, static_cast<int>(numCells)), [&](const cv::Range &range){
        cv::Mat localFilteredImage, localHMap;
        for(int i = range.start; i < range.end; i++){
            size_t r = i / numCellsW;
            size_t c = i % numCellsW;
            if(occupiedCells[i]) continue;
            size_t x = c * cellSize;
            size_t y = r * cellSize;
            if(x + cellSize >= numCols || y + cellSize >= numRows) continue;
            cv::Rect regionOfInterest(x, y, cellSize, cellSize);
            
            // Use faster blur with smaller kernel
            cv::GaussianBlur(image(regionOfInterest), localFilteredImage, cv::Size(3, 3), 0.);
            cv::cornerMinEigenVal(localFilteredImage, localHMap, 3, 3);
            
            double minVal1, maxVal1;
            cv::Point minPx1, maxPx1;
            cv::minMaxLoc(localHMap.mul(mask(regionOfInterest)), &minVal1, &maxVal1, &minPx1, &maxPx1);
            maxPx1.x += x;
            maxPx1.y += y;
            if(maxPx1.x >= roi.x && maxPx1.y >= roi.y && 
                    maxPx1.x < roi.x + roi.width && maxPx1.y < roi.y + roi.height && 
                    maxVal1 >= maxQuality_){
                primaryDetections[i].push_back(cv::Point2f(maxPx1));
                // Use atomic operations for thread safety
                cv::circle(mask, maxPx1, cellSizeHalf, cv::Scalar(0.), -1);

                double minVal2, maxVal2;
                cv::Point minPx2, maxPx2;
                cv::minMaxLoc(localHMap.mul(mask(regionOfInterest)), &minVal2, &maxVal2, &minPx2, &maxPx2);
                maxPx2.x += x;
                maxPx2.y += y;
                if(maxPx2.x >= roi.x && maxPx2.y >= roi.y && 
                        maxPx2.x < roi.x + roi.width && maxPx2.y < roi.y + roi.height && 
                       maxVal2 >= maxQuality_){
                    secondaryDetections[i].push_back(cv::Point2f(maxPx2));
                }
            }
        }
    }, numThreads);

    size_t primaryCount = 0;
    for(const auto &detection : primaryDetections){
        if(!detection.empty()){
            detectedPx.push_back(detection[0]);
            primaryCount++;
        }
    }

    size_t available = numCells - numOccupied;
    size_t targetCount = available * 0.9;

    if(primaryCount < available){
        for(const auto &detection : secondaryDetections){
            if(!detection.empty() && detectedPx.size() < targetCount){
                detectedPx.push_back(detection[0]);
            }
        }
    }

    if(detectedPx.size() < 0.33 * available){
        maxQuality_ *= 0.5;
    }else if(detectedPx.size() > 0.9 * available){
        maxQuality_ *= 1.5;
    }
 
    if(!detectedPx.empty()){
        cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.01);
        cv::cornerSubPix(image, detectedPx, cv::Size(3, 3), cv::Size(-1, -1), criteria);
    }

    return detectedPx;
}


std::vector<cv::Mat> FeatureExtractor::describeFeaturePoints(
    const cv::Mat &image, 
    const std::vector<cv::Point2f> &points) const{
    if(points.empty()){
        return {};
    }

    std::vector<cv::KeyPoint> keypoints;
    keypoints.reserve(points.size());
    cv::KeyPoint::convert(points, keypoints);

    cv::Mat descriptorsMatrix;
    descriptor_->compute(image, keypoints, descriptorsMatrix);

    std::vector<cv::Mat> descriptors(points.size());
    
    if(!descriptorsMatrix.empty()){
        std::unordered_map<std::string, size_t> keypointMap;
        for (size_t i = 0; i < keypoints.size(); ++i) {
            std::string key = std::to_string(int(keypoints[i].pt.x * 100)) + "_" + 
                              std::to_string(int(keypoints[i].pt.y * 100));
            keypointMap[key] = i;
        }

        for(size_t i = 0; i < points.size(); ++i){
            std::string key = std::to_string(int(points[i].x * 100)) + "_" + 
                             std::to_string(int(points[i].y * 100));
            
            auto it = keypointMap.find(key);
            if(it != keypointMap.end()){
                descriptors[i] = descriptorsMatrix.row(it->second).clone();
            }
        }
    }

    return descriptors;
}