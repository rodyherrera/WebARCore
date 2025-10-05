#pragma once

#include <vector>
#include <Eigen/Core>

namespace opengv {
    typedef Eigen::Vector3d point_t;
    typedef Eigen::Vector3d bearing_vector_t;
    typedef Eigen::Matrix3d rotation_t;
    typedef Eigen::Vector3d translation_t;
    typedef Eigen::Matrix4d transformation_t;
    typedef std::vector<point_t> points_t;
    typedef std::vector<bearing_vector_t> bearingVectors_t;

    namespace absolute_pose {
        class CentralAbsoluteAdapter;
    }
    
    namespace relative_pose {
        class CentralRelativeAdapter;
    }

    namespace triangulation {
        namespace methods {
            inline Eigen::Vector3d triangulate(
                const bearing_vector_t& /*bearing1*/,
                const bearing_vector_t& /*bearing2*/,
                const Eigen::Matrix3d& /*rotation*/,
                const Eigen::Vector3d& /*translation*/) {
                return Eigen::Vector3d::Zero();
            }

            inline Eigen::Vector3d triangulate2(
                void* /*adapter*/,
                int /*index*/) {
                return Eigen::Vector3d::Zero();
            }
        }
    }

    namespace sac {
        struct SampleConsensusModel {
            virtual ~SampleConsensusModel() = default;
            std::vector<int> inliers_;
            Eigen::Matrix4d model_coefficients_;
            
            virtual void optimizeModelCoefficients(
                const std::vector<int>& /*inliers*/,
                const Eigen::Matrix4d& /*model_coefficients*/,
                Eigen::Matrix4d& /*optimized_coefficients*/) {}
        };

        template<typename MODEL_T>
        class Ransac {
        public:
            Ransac() {}
            Ransac(MODEL_T* model) : sac_model_(model) {}
            bool computeModel() { return false; }
            bool computeModel(int /*threads*/) { return false; }
            std::vector<int> getInliers() { return inliers_; }
            
            MODEL_T* sac_model_ = nullptr;
            std::vector<int> inliers_;
            Eigen::Matrix4d model_coefficients_;
            double threshold_ = 0.0;
            int max_iterations_ = 100;
        };

        template<typename MODEL_T>
        class Lmeds {
        public:
            Lmeds() {}
            Lmeds(MODEL_T* model) : sac_model_(model) {}
            bool computeModel() { return false; }
            bool computeModel(int /*threads*/) { return false; }
            std::vector<int> getInliers() { return inliers_; }
            
            MODEL_T* sac_model_ = nullptr;
            std::vector<int> inliers_;
            Eigen::Matrix4d model_coefficients_;
            double threshold_ = 0.0;
            int max_iterations_ = 100;
        };
    }

    namespace absolute_pose {
        class CentralAbsoluteAdapter {
        public:
            CentralAbsoluteAdapter(
                const bearingVectors_t& /*bearingVectors*/,
                const points_t& /*points*/) {}
        };
    }

    namespace sac_problems {
        namespace absolute_pose {
            class AbsolutePoseSacProblem : public sac::SampleConsensusModel {
            public:
                enum { KNEIP = 0, GAO = 1, EPNP = 2 };
                
                AbsolutePoseSacProblem() {}
                AbsolutePoseSacProblem(
                    opengv::absolute_pose::CentralAbsoluteAdapter* /*adapter*/,
                    int /*algorithm*/) {}
            };
        }

        namespace relative_pose {
            class CentralRelativePoseSacProblem : public sac::SampleConsensusModel {
            public:
                enum { NISTER = 0, STEWENIUS = 1, SEVENPT = 2, EIGHTPT = 3 };
                
                CentralRelativePoseSacProblem() {}
                CentralRelativePoseSacProblem(
                    opengv::relative_pose::CentralRelativeAdapter* /*adapter*/,
                    int /*algorithm*/) {}
            };
        }
    }

    namespace relative_pose {
        class CentralRelativeAdapter {
        public:
            CentralRelativeAdapter(
                const bearingVectors_t& /*bearingVectors1*/,
                const bearingVectors_t& /*bearingVectors2*/) {}
                
            CentralRelativeAdapter(
                const bearingVectors_t& /*bearingVectors1*/,
                const bearingVectors_t& /*bearingVectors2*/,
                const translation_t& /*translation*/,
                const rotation_t& /*rotation*/) {}
        };
    }
}