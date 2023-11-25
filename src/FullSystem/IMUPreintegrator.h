#ifndef SLAM_IMUPREINTEGRATOR_H
#define SLAM_IMUPREINTEGRATOR_H

#include "util/NumType.h"
#include "sophus/se3.hpp"
#include "sophus/so3.hpp"
#include <Eigen/Dense>

namespace dso
{
    class IMUPreintegrator
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        IMUPreintegrator();

        IMUPreintegrator(const IMUPreintegrator &pre);

        // reset to initial state
        void reset();

        // incrementally update 1)delta measurements, 2)jacobians, 3)covariance matrix
        void update(const Vec3 &omega, const Vec3 &acc, const double &dt);

        // delta measurements, position/velocity/rotation(matrix)
        inline Eigen::Vector3d getDeltaP() const    // P_k+1 = P_k + V_k*dt + R_k*a_k*dt*dt/2
        {
            return _delta_P;
        }

        inline Eigen::Vector3d getDeltaV() const    // V_k+1 = V_k + R_k*a_k*dt
        {
            return _delta_V;
        }

        inline Eigen::Matrix3d getDeltaR() const   // R_k+1 = R_k*exp(w_k*dt).     NOTE: Rwc, Rwc'=Rwc*[w_body]x
        {
            return _delta_R;
        }

        inline Eigen::Matrix3d getJPBiasg() const     // position / gyro
        {
            return _J_P_Biasg;
        }

        inline Eigen::Matrix3d getJPBiasa() const     // position / acc
        {
            return _J_P_Biasa;
        }

        inline Eigen::Matrix3d getJVBiasg() const     // velocity / gyro
        {
            return _J_V_Biasg;
        }

        inline Eigen::Matrix3d getJVBiasa() const     // velocity / acc
        {
            return _J_V_Biasa;
        }

        inline Eigen::Matrix3d getJRBiasg() const  // rotation / gyro
        {
            return _J_R_Biasg;
        }

        // noise covariance propagation of delta measurements
        // note: the order is position-velocity-rotation here
        inline Mat99 getCovPVPhi() const
        {
            return _cov_P_V_Phi;
        }

        inline double getDeltaTime() const
        {
            return _delta_time;
        }

        // skew-symmetric matrix
        static Mat33 skew(const Vec3 &v)
        {
            return Sophus::SO3d::hat(v);
        }

        // exponential map from Vec3 to mat3x3 (Rodrigues formula)
        static Mat33 Expmap(const Vec3 &v)
        {
            return Sophus::SO3d::exp(v).matrix();
        }

        // right jacobian of SO(3)
        static Mat33 JacobianR(const Vec3 &w)
        {
            Mat33 Jr = Mat33::Identity();
            double theta = w.norm();

            if (theta < 1e-5)
                return Jr;
            else
            {
                Vec3 k = w.normalized();
                Mat33 K = skew(k);
                Jr = sin(theta) / theta * Mat33::Identity() +
                        (1 - sin(theta) / theta) * k * k.transpose() -
                        (1 - cos(theta)) / theta * K;
            }
            return Jr;
        }

        // right jacobian inverse of SO3
        static Mat33 JacobianRInv(const Vec3 &w)
        {
            Mat33 Jrinv = Mat33::Identity();
            double theta = w.norm();

            if (theta < 1e-5)
                return Jrinv;
            else
            {
                Vec3 k = w.normalized();
                Mat33 K = skew(k);
                double cot = cos(theta / 2) / sin(theta / 2);
                Jrinv = theta / 2 * cot * Mat33::Identity() +
                        (1 - theta / 2 * cot) * k * k.transpose() +
                        theta / 2 * K;
            }

            return Jrinv;
        }

        // left jacobian of SO(3), Jl(x) = Jr(-x)
        static Mat33 JacobianL(const Vec3 &w)
        {
            return JacobianR(-w);
        }

        // left jacobian inverse
        static Mat33 JacobianLInv(const Vec3 &w)
        {
            return JacobianRInv(-w);
        }

        inline Sophus::Quaterniond normalizeRotationQ(const Sophus::Quaterniond &r)
        {
            Sophus::Quaterniond _r(r);
            if (_r.w() < 0)
                _r.coeffs() *= -1;
            return _r.normalized();
        }

        inline Mat33 normalizeRotationM(const Mat33 &R)
        {
            Sophus::Quaterniond qr(R);
            return normalizeRotationQ(qr).toRotationMatrix();
        }

    private:

        // delta measurements, position/velocity/rotation(matrix)
        Eigen::Vector3d _delta_P;    // P_k+1 = P_k + V_k*dt + R_k*a_k*dt*dt/2
        Eigen::Vector3d _delta_V;    // V_k+1 = V_k + R_k*a_k*dt
        Eigen::Matrix3d _delta_R;    // R_k+1 = R_k*exp(w_k*dt). ote: Rwc, Rwc'=Rwc*[w_body]x

        // jacobian of delta measurements w.r.t bias of gyro/acc
        Eigen::Matrix3d _J_P_Biasg;     // position / gyro
        Eigen::Matrix3d _J_P_Biasa;     // position / acc
        Eigen::Matrix3d _J_V_Biasg;     // velocity / gyro
        Eigen::Matrix3d _J_V_Biasa;     // velocity / acc
        Eigen::Matrix3d _J_R_Biasg;     // rotation / gyro

        std::vector<Vec3> _gyr_data;    // 陀螺仪数据
        std::vector<Vec3> _acc_data;    // 加速度数据

        // noise covariance propagation of delta measurements
        Mat99 _cov_P_V_Phi;

        double _delta_time;
    };
}
#endif