#include "imu_gps_localizer/imu_processor.h"

#include <glog/logging.h>
#include <Eigen/Dense>

#include "imu_gps_localizer/utils.h"

namespace ImuGpsLocalization {

ImuProcessor::ImuProcessor(const double acc_noise, const double gyro_noise,
                           const double acc_bias_noise, const double gyro_bias_noise,
                           const Eigen::Vector3d& gravity)
    : acc_noise_(acc_noise), gyro_noise_(gyro_noise), 
      acc_bias_noise_(acc_bias_noise), gyro_bias_noise_(gyro_bias_noise),
      gravity_(gravity) { }

// 预测，采用帧的imu数据和当前imu数据进行航迹推算
void ImuProcessor::Predict(const ImuDataPtr last_imu, const ImuDataPtr cur_imu, State* state) {
    // Time.两次时间戳
    const double delta_t = cur_imu->timestamp - last_imu->timestamp;
    const double delta_t2 = delta_t * delta_t;

    // Set last state.记录上刻状态
    State last_state = *state;

    // Acc and gyro.
    // 矫正加速度和角速度零偏
    // 获取平均加速度和角速度
    // 假设a为在delta_t中均匀变化，故delta_t间隔间的a应为0.5×（a_last+a_curr）
    // 其中acc_bias为0偏
    const Eigen::Vector3d acc_unbias = 0.5 * (last_imu->acc + cur_imu->acc) - last_state.acc_bias;
    const Eigen::Vector3d gyro_unbias = 0.5 * (last_imu->gyro + cur_imu->gyro) - last_state.gyro_bias;

    // Normal state. 
    // Using P58. of "Quaternion kinematics for the error-state Kalman Filter".
    // 预测pose， pose = pose + v*t + 0.5*a*t*t, 其中a = R*a_sensor*+gravity
    // gravity_ 重力加速度修正，需要减去本身存在的加速度
    // 测量的的加速度根据旋转矩阵转换到世界坐标系下
    state->G_p_I = last_state.G_p_I + last_state.G_v_I * delta_t + 
                   0.5 * (last_state.G_R_I * acc_unbias + gravity_) * delta_t2;
    // vel = vel + a*t, 其中a = R*a_sensor*+gravity
    state->G_v_I = last_state.G_v_I + (last_state.G_R_I * acc_unbias + gravity_) * delta_t;
    // 角度变换量， 更新最新的旋转矩阵
    const Eigen::Vector3d delta_angle_axis = gyro_unbias * delta_t;
    // 相当于滤波，当更新偏转一定阈值时，才可更新姿态角
    if (delta_angle_axis.norm() > 1e-12) {
        // 将旋转向量转换成旋转矩阵，然后进行旋转
        state->G_R_I = last_state.G_R_I * Eigen::AngleAxisd(delta_angle_axis.norm(), delta_angle_axis.normalized()).toRotationMatrix();
    }
    // Error-state. Not needed.
    // 转移方程无需error state，但是协方差需要

    // Covariance of the error-state.   
    // 更新状态协方差矩阵

    // 转移矩阵 x = F*x+u
    Eigen::Matrix<double, 15, 15> Fx = Eigen::Matrix<double, 15, 15>::Identity();
    // pose
    Fx.block<3, 3>(0, 3)   = Eigen::Matrix3d::Identity() * delta_t;
    // v= at= R*a*t,
    // R的偏导数
    Fx.block<3, 3>(3, 6)   = - state->G_R_I * GetSkewMatrix(acc_unbias) * delta_t;
    // a的偏导数
    Fx.block<3, 3>(3, 9)   = - state->G_R_I * delta_t;
    // R = R+ grac*delta_t;
    // R的偏导数
    if (delta_angle_axis.norm() > 1e-12) {
        Fx.block<3, 3>(6, 6) = Eigen::AngleAxisd(delta_angle_axis.norm(), delta_angle_axis.normalized()).toRotationMatrix().transpose();
    } else {
        Fx.block<3, 3>(6, 6).setIdentity();
    }
    // R = R+ grac*delta_t;
    // 这是grac的偏导数
    Fx.block<3, 3>(6, 12)  = - Eigen::Matrix3d::Identity() * delta_t;
    // acc_bias 和 gyro_bias的偏导数为单位矩阵，
    // acc = acc_last;
    // w = w_last;

    // 误差状态传递函数对干扰项的导数。
    Eigen::Matrix<double, 15, 12> Fi = Eigen::Matrix<double, 15, 12>::Zero();
    Fi.block<12, 12>(3, 0) = Eigen::Matrix<double, 12, 12>::Identity();

    // 误差项主要是传感器值存在误差
    Eigen::Matrix<double, 12, 12> Qi = Eigen::Matrix<double, 12, 12>::Zero();
    Qi.block<3, 3>(0, 0) = delta_t2 * acc_noise_ * Eigen::Matrix3d::Identity();
    Qi.block<3, 3>(3, 3) = delta_t2 * gyro_noise_ * Eigen::Matrix3d::Identity();
    Qi.block<3, 3>(6, 6) = delta_t * acc_bias_noise_ * Eigen::Matrix3d::Identity();
    Qi.block<3, 3>(9, 9) = delta_t * gyro_bias_noise_ * Eigen::Matrix3d::Identity();

    // 更新协方差矩阵
    state->cov = Fx * last_state.cov * Fx.transpose() + Fi * Qi * Fi.transpose();

    // Time and imu.
    state->timestamp = cur_imu->timestamp;
    state->imu_data_ptr = cur_imu;
}

}  // namespace ImuGpsLocalization