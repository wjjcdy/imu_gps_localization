#include "imu_gps_localizer/initializer.h"

#include <Eigen/Dense>
#include <glog/logging.h>

#include "imu_gps_localizer/utils.h"

namespace ImuGpsLocalization {

Initializer::Initializer(const Eigen::Vector3d& init_I_p_Gps) 
    : init_I_p_Gps_(init_I_p_Gps) { }

// imu buffer, 将imudata 按队列放入buffer中，剔除最旧的数据
void Initializer::AddImuData(const ImuDataPtr imu_data_ptr) {
    imu_buffer_.push_back(imu_data_ptr);

    if (imu_buffer_.size() > kImuDataBufferLength) {
        imu_buffer_.pop_front();
    }
}

// gps buffer, 同理将gpsdata 放入buffer中，剔除最旧的数据
// 初始化时，需要imu的的buffer提前缓存一定数量，
// 其目的是用gps的数据来校准imu数据的
bool Initializer::AddGpsPositionData(const GpsPositionDataPtr gps_data_ptr, State* state) {
    if (imu_buffer_.size() < kImuDataBufferLength) {
        LOG(WARNING) << "[AddGpsPositionData]: No enought imu data!";
        return false;
    }

    // 提取IMU最新数据
    const ImuDataPtr last_imu_ptr = imu_buffer_.back();
    // TODO: synchronize all sensors.
    // 判断最新的imu数据和gps的数据时间误差需要在0.5s以内
    if (std::abs(gps_data_ptr->timestamp - last_imu_ptr->timestamp) > 0.5) {
        LOG(ERROR) << "[AddGpsPositionData]: Gps and imu timestamps are not synchronized!";
        return false;
    }

    // 设置状态参数
    // Set timestamp and imu date.
    state->timestamp = last_imu_ptr->timestamp;
    state->imu_data_ptr = last_imu_ptr;

    // Set initial mean. 初始位置
    state->G_p_I.setZero();

    // We have no information to set initial velocity. 
    // So, just set it to zero and given big covariance.
    // 初始速度
    state->G_v_I.setZero();

    // We can use the direction of gravity to set roll and pitch. 
    // But, we cannot set the yaw. 
    // So, we set yaw to zero and give it a big covariance.
    // imu可以根据重力直接获取roll和pitch，而yaw未知，赋个初始值
    // 上电时根据acc获取初始姿态角，其中航角初始为0
    if (!ComputeG_R_IFromImuData(&state->G_R_I)) {
        LOG(WARNING) << "[AddGpsPositionData]: Failed to compute G_R_I!";
        return false;
    }

    // 设置初始状态
    // Set bias to zero. 零偏默认为0;
    state->acc_bias.setZero();
    state->gyro_bias.setZero();

    // Set covariance.
    state->cov.setZero();
    state->cov.block<3, 3>(0, 0) = 100. * Eigen::Matrix3d::Identity(); // position std: 10 m
    state->cov.block<3, 3>(3, 3) = 100. * Eigen::Matrix3d::Identity(); // velocity std: 10 m/s
    // roll pitch std 10 degree.
    state->cov.block<2, 2>(6, 6) = 10. * kDegreeToRadian * 10. * kDegreeToRadian * Eigen::Matrix2d::Identity();
    state->cov(8, 8)             = 100. * kDegreeToRadian * 100. * kDegreeToRadian; // yaw std: 100 degree.
    // Acc bias.
    state->cov.block<3, 3>(9, 9) = 0.0004 * Eigen::Matrix3d::Identity();
    // Gyro bias.
    state->cov.block<3, 3>(12, 12) = 0.0004 * Eigen::Matrix3d::Identity();

    return true;
}

//从imu数据中获取姿态角
bool Initializer::ComputeG_R_IFromImuData(Eigen::Matrix3d* G_R_I) {
    // Compute mean and std of the imu buffer.
    Eigen::Vector3d sum_acc(0., 0., 0.);
    // 统计imu buffer中 均值 
    for (const auto imu_data : imu_buffer_) {
        sum_acc += imu_data->acc;
    }
    const Eigen::Vector3d mean_acc = sum_acc / (double)imu_buffer_.size();

    Eigen::Vector3d sum_err2(0., 0., 0.);
    // 误差平方和
    for (const auto imu_data : imu_buffer_) {
        sum_err2 += (imu_data->acc - mean_acc).cwiseAbs2();
    }
    // 计算加速度协方差（方差开方）
    const Eigen::Vector3d std_acc = (sum_err2 / (double)imu_buffer_.size()).cwiseSqrt();

    // 判断加速度误差大小，过大则抛弃
    if (std_acc.maxCoeff() > kAccStdLimit) {
        LOG(WARNING) << "[ComputeG_R_IFromImuData]: Too big acc std: " << std_acc.transpose();
        return false;
    }

    // Compute rotation.
    // Please refer to 
    // https://github.com/rpng/open_vins/blob/master/ov_core/src/init/InertialInitializer.cpp
    
    // Three axises of the ENU frame in the IMU frame.
    // z-axis.
    // 假设理论上无旋转时则加速度归一化后和大地坐标系一致，应为（0,0,1）‘
    // 旋转后，加速度归一化结果，同时也为旋转矩阵中的z轴向量
    const Eigen::Vector3d& z_axis = mean_acc.normalized(); 

    // x-axis.横滚一维矩阵， 假设初始偏航角为0度
    // ??????表示不懂公式如何来的
    Eigen::Vector3d x_axis = 
        Eigen::Vector3d::UnitX() - z_axis * z_axis.transpose() * Eigen::Vector3d::UnitX();
    x_axis.normalize();

    // y-axis.俯仰一维矩阵
    Eigen::Vector3d y_axis = z_axis.cross(x_axis);
    y_axis.normalize();

    Eigen::Matrix3d I_R_G;
    I_R_G.block<3, 1>(0, 0) = x_axis;
    I_R_G.block<3, 1>(0, 1) = y_axis;
    I_R_G.block<3, 1>(0, 2) = z_axis;

    // 转置
    *G_R_I = I_R_G.transpose();

    return true;
}

}  // namespace ImuGpsLocalization