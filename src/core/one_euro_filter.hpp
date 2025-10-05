#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <chrono>
#include <cmath>

/**
 * One Euro Filter - Low-pass filter with adaptive cutoff frequency
 * Reference: "1€ Filter: A Simple Speed-based Low-pass Filter for Noisy Input in Interactive Systems"
 * Casiez et al. (CHI 2012)
 * 
 * Perfect for smoothing AR/VR tracking data with minimal latency
 */
class OneEuroFilter {
public:
    /**
     * @param minCutoff Minimum cutoff frequency (lower = more smoothing)
     * @param beta Speed coefficient (higher = more reactive to fast movements)
     * @param dCutoff Cutoff frequency for derivative (typically 1.0)
     */
    OneEuroFilter(double minCutoff = 1.0, double beta = 0.007, double dCutoff = 1.0)
        : minCutoff_(minCutoff)
        , beta_(beta)
        , dCutoff_(dCutoff)
        , initialized_(false)
        , lastTime_(0.0)
    {}

    void reset() {
        initialized_ = false;
        lastTime_ = 0.0;
    }

    double filter(double value, double timestamp) {
        if (!initialized_) {
            initialized_ = true;
            lastValue_ = value;
            lastDerivative_ = 0.0;
            lastTime_ = timestamp;
            return value;
        }

        double dt = timestamp - lastTime_;
        if (dt <= 0.0) dt = 0.001; // Prevent division by zero

        // Compute derivative
        double derivative = (value - lastValue_) / dt;
        double smoothedDerivative = exponentialSmoothing(derivative, lastDerivative_, alpha(dt, dCutoff_));

        // Compute cutoff frequency based on speed
        double cutoff = minCutoff_ + beta_ * std::abs(smoothedDerivative);

        // Filter the value
        double smoothedValue = exponentialSmoothing(value, lastValue_, alpha(dt, cutoff));

        // Update state
        lastValue_ = smoothedValue;
        lastDerivative_ = smoothedDerivative;
        lastTime_ = timestamp;

        return smoothedValue;
    }

private:
    double alpha(double dt, double cutoff) const {
        double tau = 1.0 / (2.0 * M_PI * cutoff);
        return 1.0 / (1.0 + tau / dt);
    }

    double exponentialSmoothing(double value, double lastValue, double alpha) const {
        return alpha * value + (1.0 - alpha) * lastValue;
    }

    double minCutoff_;
    double beta_;
    double dCutoff_;
    bool initialized_;
    double lastTime_;
    double lastValue_;
    double lastDerivative_;
};

/**
 * One Euro Filter for 3D vectors
 */
class OneEuroFilterVec3 {
public:
    OneEuroFilterVec3(double minCutoff = 1.0, double beta = 0.007, double dCutoff = 1.0)
        : filterX_(minCutoff, beta, dCutoff)
        , filterY_(minCutoff, beta, dCutoff)
        , filterZ_(minCutoff, beta, dCutoff)
    {}

    void reset() {
        filterX_.reset();
        filterY_.reset();
        filterZ_.reset();
    }

    Eigen::Vector3d filter(const Eigen::Vector3d& value, double timestamp) {
        return Eigen::Vector3d(
            filterX_.filter(value.x(), timestamp),
            filterY_.filter(value.y(), timestamp),
            filterZ_.filter(value.z(), timestamp)
        );
    }

private:
    OneEuroFilter filterX_;
    OneEuroFilter filterY_;
    OneEuroFilter filterZ_;
};

/**
 * One Euro Filter for quaternions (rotation smoothing)
 */
class OneEuroFilterQuat {
public:
    OneEuroFilterQuat(double minCutoff = 1.0, double beta = 0.007, double dCutoff = 1.0)
        : filterW_(minCutoff, beta, dCutoff)
        , filterX_(minCutoff, beta, dCutoff)
        , filterY_(minCutoff, beta, dCutoff)
        , filterZ_(minCutoff, beta, dCutoff)
    {}

    void reset() {
        filterW_.reset();
        filterX_.reset();
        filterY_.reset();
        filterZ_.reset();
    }

    Eigen::Quaterniond filter(const Eigen::Quaterniond& value, double timestamp) {
        // Ensure quaternion is normalized
        Eigen::Quaterniond q = value.normalized();
        
        Eigen::Quaterniond filtered(
            filterW_.filter(q.w(), timestamp),
            filterX_.filter(q.x(), timestamp),
            filterY_.filter(q.y(), timestamp),
            filterZ_.filter(q.z(), timestamp)
        );
        
        // Normalize result
        return filtered.normalized();
    }

private:
    OneEuroFilter filterW_;
    OneEuroFilter filterX_;
    OneEuroFilter filterY_;
    OneEuroFilter filterZ_;
};

/**
 * One Euro Filter for SE3 poses (position + rotation)
 */
class OneEuroFilterSE3 {
public:
    /**
     * @param minCutoffPos Minimum cutoff for position (lower = smoother, higher = more responsive)
     * @param minCutoffRot Minimum cutoff for rotation
     * @param beta Speed coefficient for adaptivity
     */
    OneEuroFilterSE3(double minCutoffPos = 1.0, double minCutoffRot = 1.0, 
                     double beta = 0.007, double dCutoff = 1.0)
        : positionFilter_(minCutoffPos, beta, dCutoff)
        , rotationFilter_(minCutoffRot, beta, dCutoff)
    {}

    void reset() {
        positionFilter_.reset();
        rotationFilter_.reset();
    }

    void filter(Sophus::SE3d& pose, double timestamp) {
        // Filter translation
        Eigen::Vector3d filteredTranslation = positionFilter_.filter(pose.translation(), timestamp);
        
        // Filter rotation (via quaternion)
        Eigen::Quaterniond quat(pose.rotationMatrix());
        Eigen::Quaterniond filteredQuat = rotationFilter_.filter(quat, timestamp);
        
        // Reconstruct SE3 pose
        pose = Sophus::SE3d(filteredQuat, filteredTranslation);
    }

private:
    OneEuroFilterVec3 positionFilter_;
    OneEuroFilterQuat rotationFilter_;
};
