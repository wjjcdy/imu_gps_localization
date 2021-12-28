#include "imu_gps_localizer/imu_gps_localizer.h"

#include <glog/logging.h>

#include "imu_gps_localizer/utils.h"

namespace ImuGpsLocalization {

ImuGpsLocalizer::ImuGpsLocalizer(const double acc_noise, const double gyro_noise,
                                 const double acc_bias_noise, const double gyro_bias_noise,
                                 const Eigen::Vector3d& I_p_Gps) 
    : initialized_(false){
    // 上电用于给出初始状态的，其中pose直接为gps，而roll和pitch为imu中acc获取
    // 而yaw则人为给定航向角
    initializer_ = std::make_unique<Initializer>(I_p_Gps);
    // imu data 融合处理
    // 其中-9.8为重力加速度修正值
    imu_processor_ = std::make_unique<ImuProcessor>(acc_noise, gyro_noise, 
                                                    acc_bias_noise, gyro_bias_noise,
                                                    Eigen::Vector3d(0., 0., -9.81007));
    // gps 处理
    gps_processor_ = std::make_unique<GpsProcessor>(I_p_Gps);
}

// imu 融合处理
bool ImuGpsLocalizer::ProcessImuData(const ImuDataPtr imu_data_ptr, State* fused_state) {
    // 未初始化需要先进行初始化，即获取初始状态
    if (!initialized_) {
        initializer_->AddImuData(imu_data_ptr);
        return false;
    }
    
    // Predict. imu进行预测
    imu_processor_->Predict(state_.imu_data_ptr, imu_data_ptr, &state_);

    // Convert ENU state to lla.
    ConvertENUToLLA(init_lla_, state_.G_p_I, &(state_.lla));
    *fused_state = state_;
    return true;
}

bool ImuGpsLocalizer::ProcessGpsPositionData(const GpsPositionDataPtr gps_data_ptr) {
    if (!initialized_) {
        if (!initializer_->AddGpsPositionData(gps_data_ptr, &state_)) {
            return false;
        }

        // Initialize the initial gps point used to convert lla to ENU.
        init_lla_ = gps_data_ptr->lla;
        
        initialized_ = true;

        LOG(INFO) << "[ProcessGpsPositionData]: System initialized!";
        return true;
    }

    // Update.
    gps_processor_->UpdateStateByGpsPosition(init_lla_, gps_data_ptr, &state_);

    return true;
}

}  // namespace ImuGpsLocalization