#pragma once
#include "ceres_parametrization.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <sophus/se3.hpp>
#include <cassert>

namespace DirectSE3
{

// Utilidad para verificar dimensiones en tiempo de compilación
constexpr int kCalibDim = 4;
constexpr int kPoseDim = 7;
constexpr int kPointDim = 3;

static_assert(kCalibDim == 4, "Calibración debe tener 4 parámetros");
static_assert(kPoseDim == 7, "Pose debe tener 7 parámetros (t + q)");
static_assert(kPointDim == 3, "Punto debe ser 3D");

bool ReprojectionErrorKSE3XYZ::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    // Calibración: [fx, fy, cx, cy]
    const Eigen::Map<const Eigen::Vector4d> lcalib(parameters[0]);
    // Pose: [tx, ty, tz, qw, qx, qy, qz]
    const Eigen::Map<const Eigen::Vector3d> twc(parameters[1]);
    const Eigen::Map<const Eigen::Quaterniond> qwc(parameters[1] + 3);
    Sophus::SE3d Twc(qwc, twc);
    Sophus::SE3d Tcw = Twc.inverse();
    // Punto en mundo: [x, y, z]
    const Eigen::Map<const Eigen::Vector3d> wpt(parameters[2]);
    // Proyección
    Eigen::Vector3d lcampt = Tcw * wpt;
    const double linvz = 1.0 / lcampt.z();
    Eigen::Vector2d pred;
    pred << lcalib(0) * lcampt.x() * linvz + lcalib(2),
            lcalib(1) * lcampt.y() * linvz + lcalib(3);

    // Residuals
    Eigen::Map<Eigen::Vector2d> werr(residuals);
    werr = sqrt_info_ * (pred - unpx_);

    // Actualiza chi2 y bandera de profundidad positiva
    chi2err_ = werr.squaredNorm();
    isDepthPositive_ = lcampt.z() > 0;

    // Jacobians
    if (jacobians)
    {
        const double linvz2 = linvz * linvz;
        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_lcam;
        J_lcam << linvz * lcalib(0), 0., -lcampt.x() * linvz2 * lcalib(0),
                  0., linvz * lcalib(1), -lcampt.y() * linvz2 * lcalib(1);

        Eigen::Matrix<double, 2, 3> J_lRcw;
        if (jacobians[1] || jacobians[2])
            J_lRcw.noalias() = J_lcam * Tcw.rotationMatrix();

        // Calibración
        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J_lcalib(jacobians[0]);
            J_lcalib.setZero();
            J_lcalib(0, 0) = lcampt.x() * linvz;
            J_lcalib(0, 2) = 1.;
            J_lcalib(1, 1) = lcampt.y() * linvz;
            J_lcalib(1, 3) = 1.;
            J_lcalib = sqrt_info_ * J_lcalib.eval();
        }
        // Pose
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_se3pose(jacobians[1]);
            J_se3pose.setZero();
            J_se3pose.block<2, 3>(0, 0)        = -J_lRcw;
            J_se3pose.block<2, 3>(0, 3).noalias() = J_lRcw * Sophus::SO3d::hat(wpt);
            J_se3pose = sqrt_info_ * J_se3pose.eval();
        }
        // Punto
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> J_wpt(jacobians[2]);
            J_wpt.setZero();
            J_wpt = sqrt_info_ * J_lRcw.eval();
        }
    }
    return true;
}

bool ReprojectionErrorSE3::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    // Pose: [tx, ty, tz, qw, qx, qy, qz]
    const Eigen::Map<const Eigen::Vector3d> twc(parameters[0]);
    const Eigen::Map<const Eigen::Quaterniond> qwc(parameters[0] + 3);
    Sophus::SE3d Twc(qwc, twc);
    Sophus::SE3d Tcw = Twc.inverse();

    Eigen::Vector3d lcampt = Tcw * wpt_;
    const double linvz = 1.0 / lcampt.z();
    Eigen::Vector2d pred;
    pred << fx_ * lcampt.x() * linvz + cx_,
            fy_ * lcampt.y() * linvz + cy_;

    Eigen::Map<Eigen::Vector2d> werr(residuals);
    werr = sqrt_info_ * (pred - unpx_);
    chi2err_ = werr.squaredNorm();
    isDepthPositive_ = lcampt.z() > 0;

    if (jacobians && jacobians[0])
    {
        const double linvz2 = linvz * linvz;
        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_lcam;
        J_lcam << linvz * fx_, 0., -lcampt.x() * linvz2 * fx_,
                  0., linvz * fy_, -lcampt.y() * linvz2 * fy_;

        Eigen::Matrix<double, 2, 3> J_lRcw = J_lcam * Tcw.rotationMatrix();

        Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_se3pose(jacobians[0]);
        J_se3pose.setZero();
        J_se3pose.block<2, 3>(0, 0)        = -J_lRcw;
        J_se3pose.block<2, 3>(0, 3).noalias() = J_lRcw * Sophus::SO3d::hat(wpt_);
        J_se3pose = sqrt_info_ * J_se3pose.eval();
    }
    return true;
}

bool ReprojectionErrorKSE3AnchInvDepth::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    // Calibración: [fx, fy, cx, cy]
    const Eigen::Map<const Eigen::Vector4d> lcalib(parameters[0]);
    // Pose ancla y actual: [tx, ty, tz, qw, qx, qy, qz]
    const Eigen::Map<const Eigen::Vector3d> twanch(parameters[1]);
    const Eigen::Map<const Eigen::Quaterniond> qwanch(parameters[1] + 3);
    const Eigen::Map<const Eigen::Vector3d> twc(parameters[2]);
    const Eigen::Map<const Eigen::Quaterniond> qwc(parameters[2] + 3);
    Sophus::SE3d Twanch(qwanch, twanch);
    Sophus::SE3d Twc(qwc, twc);
    Sophus::SE3d Tcw = Twc.inverse();
    // Inverse depth
    const double zanch = 1.0 / parameters[3][0];

    // Matriz de calibración e inversa
    Eigen::Matrix3d K;
    K << lcalib(0), 0., lcalib(2),
         0., lcalib(1), lcalib(3),
         0., 0., 1.;
    const Eigen::Matrix3d invK = K.inverse();

    // Punto en el ancla y mundo
    const Eigen::Vector3d anchpt = zanch * invK * anchpx_;
    const Eigen::Vector3d wpt    = Twanch * anchpt;
    Eigen::Vector3d lcampt       = Tcw * wpt;
    const double linvz = 1.0 / lcampt.z();

    Eigen::Vector2d pred;
    pred << lcalib(0) * lcampt.x() * linvz + lcalib(2),
            lcalib(1) * lcampt.y() * linvz + lcalib(3);

    Eigen::Map<Eigen::Vector2d> werr(residuals);
    werr = sqrt_info_ * (pred - unpx_);

    chi2err_ = werr.squaredNorm();
    isDepthPositive_ = lcampt.z() > 0;

    if (jacobians)
    {
        const double linvz2 = linvz * linvz;
        Eigen::Matrix<double, 2, 3, Eigen::RowMajor> J_lcam;
        J_lcam << linvz * lcalib(0), 0., -lcampt.x() * linvz2 * lcalib(0),
                  0., linvz * lcalib(1), -lcampt.y() * linvz2 * lcalib(1);

        Eigen::Matrix<double, 2, 3> J_lRcw;
        Eigen::Matrix3d Rcw = Tcw.rotationMatrix();
        if (jacobians[1] || jacobians[2] || jacobians[3])
            J_lRcw.noalias() = J_lcam * Rcw;

        // Jacobiano w.r.t. calibración
        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 4, Eigen::RowMajor>> J_lcalib(jacobians[0]);
            J_lcalib.setZero();
            // Derivadas explícitas respecto a fx, fy, cx, cy
            // d pred_x / d fx = lcampt.x() * linvz
            // d pred_x / d cx = 1
            // d pred_y / d fy = lcampt.y() * linvz
            // d pred_y / d cy = 1
            J_lcalib(0, 0) = lcampt.x() * linvz;
            J_lcalib(0, 2) = 1.;
            J_lcalib(1, 1) = lcampt.y() * linvz;
            J_lcalib(1, 3) = 1.;
            J_lcalib = sqrt_info_ * J_lcalib.eval();
        }
        // Jacobiano w.r.t. pose del ancla
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_se3anch(jacobians[1]);
            J_se3anch.setZero();
            Eigen::Matrix3d skew_wpt = Sophus::SO3d::hat(wpt);
            J_se3anch.block<2, 3>(0, 0)        = J_lRcw;
            J_se3anch.block<2, 3>(0, 3).noalias() = -J_lRcw * skew_wpt;
            J_se3anch = sqrt_info_ * J_se3anch.eval();
        }
        // Jacobiano w.r.t. pose de la cámara
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_se3pose(jacobians[2]);
            J_se3pose.setZero();
            Eigen::Matrix3d skew_wpt = Sophus::SO3d::hat(wpt);
            J_se3pose.block<2, 3>(0, 0)        = -J_lRcw;
            J_se3pose.block<2, 3>(0, 3).noalias() = J_lRcw * skew_wpt;
            J_se3pose = sqrt_info_ * J_se3pose.eval();
        }
        // Jacobiano w.r.t. inverse depth
        if (jacobians[3])
        {
            Eigen::Map<Eigen::Vector2d> J_invpt(jacobians[3]);
            // d anchpt / d lambda = -1/lambda^2 * invK * anchpx_
            const double inv_lambda2 = parameters[3][0] * parameters[3][0];
            Eigen::Vector3d d_anchpt_d_lambda = -1.0 / inv_lambda2 * invK * anchpx_;
            // d wpt / d lambda = Twanch.so3() * d_anchpt_d_lambda
            Eigen::Vector3d d_wpt_d_lambda = Twanch.rotationMatrix() * d_anchpt_d_lambda;
            J_invpt.noalias() = J_lRcw * d_wpt_d_lambda;
            J_invpt = sqrt_info_ * J_invpt.eval();
        }
    }
    return true;
}

} // namespace DirectSE3