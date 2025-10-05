#include "utils.hpp"

void Utils::toPoseArray(Sophus::SE3d Twc, float *pose)
{
    // Use Eigen::Map for efficient direct memory mapping (vectorized operations)
    Eigen::Matrix4f mat = Twc.matrix().cast<float>();
    Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(pose, 4, 4) = mat;
}

void Utils::toPoseArray(cv::Mat mat, float *pose)
{
    // Use Eigen::Map for efficient conversion avoiding multiple at<>() calls
    Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> mat_eigen(
        reinterpret_cast<float*>(mat.data), 4, 4);
    Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(pose, 4, 4) = mat_eigen;
}

void Utils::toPoseMat(Sophus::SE3d Twc, cv::Mat &pose)
{
    // Use Eigen::Map for efficient direct memory mapping to cv::Mat
    Eigen::Matrix4f mat = Twc.matrix().cast<float>();
    Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>>(
        reinterpret_cast<float*>(pose.data), 4, 4) = mat;
}