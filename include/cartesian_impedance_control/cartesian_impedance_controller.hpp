// Copyright (c) 2021 Franka Emika GmbH
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <unistd.h>
#include <thread>
#include <chrono>         

#include "cartesian_impedance_control/user_input_server.hpp"

#include <rclcpp/rclcpp.hpp>
#include "rclcpp/subscription.hpp"

#include <Eigen/Dense>
#include <Eigen/Eigen>
#include <boost/algorithm/clamp.hpp>

#include <controller_interface/controller_interface.hpp>

#include <franka/model.h>
#include <franka/robot.h>
#include <franka/robot_state.h>

#include "franka_hardware/franka_hardware_interface.hpp"
#include <franka_hardware/model.hpp>

#include "franka_msgs/msg/franka_robot_state.hpp"
#include "franka_msgs/msg/errors.hpp"
#include "messages_fr3/srv/set_pose.hpp"
#include "messages_fr3/msg/array2d.hpp"
#include "messages_fr3/msg/irregular_dist_array.hpp"
#include "geometry_msgs/msg/wrench_stamped.hpp"

#include "franka_semantic_components/franka_robot_model.hpp"
#include "franka_semantic_components/franka_robot_state.hpp"

#include <iostream>
#include <fstream>

#define IDENTITY Eigen::MatrixXd::Identity(6, 6)

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
using Vector7d = Eigen::Matrix<double, 7, 1>;

namespace cartesian_impedance_control {

class CartesianImpedanceController : public controller_interface::ControllerInterface {
public:
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;

  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;

  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  controller_interface::CallbackReturn on_init() override;

  controller_interface::CallbackReturn on_configure(
      const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_activate(
      const rclcpp_lifecycle::State& previous_state) override;

  controller_interface::CallbackReturn on_deactivate(
      const rclcpp_lifecycle::State& previous_state) override;

    void setPose(const std::shared_ptr<messages_fr3::srv::SetPose::Request> request, 
    std::shared_ptr<messages_fr3::srv::SetPose::Response> response);
      

 private:
    //Nodes
    rclcpp::Subscription<franka_msgs::msg::FrankaRobotState>::SharedPtr franka_state_subscriber = nullptr;
    rclcpp::Service<messages_fr3::srv::SetPose>::SharedPtr pose_srv_;
    // rclcpp::Subscription<messages_fr3::msg::Array2d>::SharedPtr repulsion_subscriber = nullptr;
    // rclcpp::Subscription<messages_fr3::msg::IrregularDistArray>::SharedPtr repulsion_subscriber = nullptr;
    rclcpp::Subscription<messages_fr3::msg::Array2d>::SharedPtr repulsion_Ipot_subscriber = nullptr;
    rclcpp::Subscription<messages_fr3::msg::Array2d>::SharedPtr repulsion_Damper_subscriber = nullptr;


    //Functions
    void topic_callback(const std::shared_ptr<franka_msgs::msg::FrankaRobotState> msg);
    // void repulsion_topic_callback(const std::shared_ptr<messages_fr3::msg::Array2d> msg);
    // void repulsion_topic_callback(const std::shared_ptr<messages_fr3::msg::IrregularDistArray> msg);
    void repulsion_topic_Ipot_callback(const std::shared_ptr<messages_fr3::msg::Array2d> msg);
    void repulsion_topic_Damper_callback(const std::shared_ptr<messages_fr3::msg::Array2d> msg);
    void updateJointStates();
    void update_stiffness_and_references();
    void arrayToMatrix(const std::array<double, 6>& inputArray, Eigen::Matrix<double, 6, 1>& resultMatrix);
    void arrayToMatrix(const std::array<double, 7>& inputArray, Eigen::Matrix<double, 7, 1>& resultMatrix);
    Eigen::Matrix<double, 7, 1> saturateTorqueRate(const Eigen::Matrix<double, 7, 1>& tau_d_calculated, const Eigen::Matrix<double, 7, 1>& tau_J_d);
    // void calcRepulsiveTorque(Eigen::Matrix<double, 6, 7> repulsive_dists);   
    // void calcRepulsiveTorque(std::vector<double> irregular_vector);
    void calcRepulsiveTorque();    
    // std::array<double, 6> convertToStdArray(geometry_msgs::msg::WrenchStamped& wrench);
    // void normalized_rep_to_rep_forces(Eigen::Array<double, 6, 7> relative_forces);
    //State vectors and matrices
    std::array<double, 7> q_subscribed;
    std::array<double, 7> tau_J_d = {0,0,0,0,0,0,0};
    std::array<double, 6> O_F_ext_hat_K = {0,0,0,0,0,0};
    Eigen::Matrix<double, 7, 1> q_subscribed_M;
    Eigen::Matrix<double, 7, 1> tau_J_d_M = Eigen::MatrixXd::Zero(7, 1);
    Eigen::Matrix<double, 6, 1> O_F_ext_hat_K_M = Eigen::MatrixXd::Zero(6,1);
    Eigen::Matrix<double, 7, 1> q_;
    Eigen::Matrix<double, 7, 1> dq_;
    Eigen::MatrixXd jacobian_transpose_pinv;  

    //Robot parameters
    const int num_joints = 7;
    const std::string state_interface_name_{"robot_state"};
    const std::string robot_name_{"panda"};
    const std::string k_robot_state_interface_name{"robot_state"};
    const std::string k_robot_model_interface_name{"robot_model"};
    franka_hardware::FrankaHardwareInterface interfaceClass;
    std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
    const double delta_tau_max_{1.0};
    const double dt = 0.001;
                
    //Impedance control variables              
    Eigen::Matrix<double, 6, 6> Lambda = IDENTITY;                                           // operational space mass matrix
    Eigen::Matrix<double, 6, 6> Sm = IDENTITY;                                               // task space selection matrix for positions and rotation
    Eigen::Matrix<double, 6, 6> Sf = Eigen::MatrixXd::Zero(6, 6);                            // task space selection matrix for forces
    Eigen::Matrix<double, 6, 6> K =  (Eigen::MatrixXd(6,6) << 250,   0,   0,   0,   0,   0,
                                                                0, 250,   0,   0,   0,   0,
                                                                0,   0, 250,   0,   0,   0,  // impedance stiffness term
                                                                0,   0,   0, 130,   0,   0,
                                                                0,   0,   0,   0, 130,   0,
                                                                0,   0,   0,   0,   0,  10).finished();

    Eigen::Matrix<double, 6, 6> D =  (Eigen::MatrixXd(6,6) <<  35,   0,   0,   0,   0,   0,
                                                                0,  35,   0,   0,   0,   0,
                                                                0,   0,  35,   0,   0,   0,  // impedance damping term
                                                                0,   0,   0,   25,   0,   0,
                                                                0,   0,   0,   0,   25,   0,
                                                                0,   0,   0,   0,   0,   6).finished();

    // Eigen::Matrix<double, 6, 6> K =  (Eigen::MatrixXd(6,6) << 250,   0,   0,   0,   0,   0,
    //                                                             0, 250,   0,   0,   0,   0,
    //                                                             0,   0, 250,   0,   0,   0,  // impedance stiffness term
    //                                                             0,   0,   0,  80,   0,   0,
    //                                                             0,   0,   0,   0,  80,   0,
    //                                                             0,   0,   0,   0,   0,  10).finished();

    // Eigen::Matrix<double, 6, 6> D =  (Eigen::MatrixXd(6,6) <<  30,   0,   0,   0,   0,   0,
    //                                                             0,  30,   0,   0,   0,   0,
    //                                                             0,   0,  30,   0,   0,   0,  // impedance damping term
    //                                                             0,   0,   0,  18,   0,   0,
    //                                                             0,   0,   0,   0,  18,   0,
    //                                                             0,   0,   0,   0,   0,   9).finished();
    Eigen::Matrix<double, 6, 6> Theta = IDENTITY;
    Eigen::Matrix<double, 6, 6> T = (Eigen::MatrixXd(6,6) <<       1,   0,   0,   0,   0,   0,
                                                                   0,   1,   0,   0,   0,   0,
                                                                   0,   0,   2.5,   0,   0,   0,  // Inertia term
                                                                   0,   0,   0,   1,   0,   0,
                                                                   0,   0,   0,   0,   1,   0,
                                                                   0,   0,   0,   0,   0,   2.5).finished();                                               // impedance inertia term

    Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_;                                 // impedance damping term
    Eigen::Matrix<double, 6, 6> cartesian_damping_target_;                                   // impedance damping term
    Eigen::Matrix<double, 6, 6> cartesian_inertia_target_;                                   // impedance damping term
    Eigen::Vector3d position_d_target_ = {0.4, 0.1, 0.6};
    Eigen::Vector3d rotation_d_target_ = {M_PI, 0.2, 0.0};
    Eigen::Quaterniond orientation_d_target_;
    Eigen::Vector3d position_d_;
    Eigen::Quaterniond orientation_d_; 
    Eigen::Matrix<double, 6, 1> F_impedance;  
    Eigen::Matrix<double, 6, 1> F_contact_des = Eigen::MatrixXd::Zero(6, 1);                 // desired contact force
    Eigen::Matrix<double, 6, 1> F_contact_target = Eigen::MatrixXd::Zero(6, 1);              // desired contact force used for filtering
    Eigen::Matrix<double, 6, 1> F_ext = Eigen::MatrixXd::Zero(6, 1);                         // external forces
    Eigen::Matrix<double, 6, 1> F_cmd = Eigen::MatrixXd::Zero(6, 1);                         // commanded contact force
    Eigen::Matrix<double, 6, 7> repulsive_dists; // = Eigen::MatrixXd::Zero(6,7);               // we assume this comes in as forces in N and Nm
    Eigen::Matrix<double, 7, 1> q_d_nullspace_;
    Eigen::Matrix<double, 6, 1> error;                                                       // pose error (6d)
    double nullspace_stiffness_{0.001};
    double nullspace_stiffness_target_{0.001};
    double impedance_limit_dist = 0.2;          // half-length of the cube outside of which spring impedance force is capped (mode 3)

    
    
    //Repulsion control variables

    double max_dist = 0.25;  // meters
    double min_dist = 0.; // meters

    // // Eigen::Array<double, 7, 1> max_admissible_moments = {87., 87., 87., 87., 12., 12., 12.};             // Nm
    // Eigen::Array<double, 7, 1> max_admissible_moments = {100., 90., 90., 90., 12., 12., 12.};  // alternative form to motivate 2ndary rotation
    // Eigen::Array<double, 7, 1> max_moments = max_admissible_moments / 3.;  // Rescale to not use max forces
    // Eigen::Array<double, 7, 1> rep_force_scaling_fraction = max_admissible_moments / max_admissible_moments.maxCoeff();  // between 0 and 1, dimensionless
    // Eigen::Array<double, 7, 1> rep_force_scaling = rep_force_scaling_fraction * 18;  // the last number is the max force in [N] that we want to apply


    ////////////////////////////////// For testing with sprig damper in cartesian space
    
    // Max moment on each joint is (87., 87., 87., 87., 12., 12., 12.) Nm
    Eigen::Array<double, 7, 1> force_allocation = {3., 3., 1.5, 1., 1., 1., 1.};                          // how to scale forces based on EE force
    double max_EE_repulsion_force = 8.;                                                         // [N]
    Eigen::Array<double, 7, 1> max_spring = (force_allocation * max_EE_repulsion_force);           // Max spring rep forces in N

    Eigen::Matrix<double, 7, 7> spring_constants = (max_spring/(max_dist-min_dist)).matrix().asDiagonal(); // [N/m]

    Eigen::Matrix<double, 7, 7> damping_constants = (12 * spring_constants.array().sqrt()).matrix(); // [N/(m/s)]

    //////////////////////////////////


    
    // // Eigen::Array<double, 7, 1> spring_constants = max_moments / (max_dist - min_dist) * 0.2;        // N, but spring constant!

    // Eigen::Array<double, 7, 1> spring_constants = rep_force_scaling / (max_dist - min_dist);        // [N/m], ensures the max force is reached at max_dist - min_dist
    // double prev_big_T = 0.;
    // //scaling factor of 1/0.5m to get a spring constant in N/m 

      Eigen::Matrix<double, 7, 1> tau_repulsion = Eigen::MatrixXd::Zero(7, 1);
      Eigen::Matrix<double, 7, 1> tau_spring_i = Eigen::MatrixXd::Zero(7, 1);
      Eigen::Matrix<double, 7, 1> tau_spring = Eigen::MatrixXd::Zero(7, 1);
      Eigen::Matrix<double, 7, 1> tau_damping = Eigen::MatrixXd::Zero(7, 1);
      Eigen::Matrix<double, 7, 1> tau_damping_i = Eigen::MatrixXd::Zero(7, 1);
      Eigen::Array<double, 6, 7> mask;
      Eigen::Matrix<double, 7, 1> unit_jt_spd = {1., 1., 1., 1., 1., 1., 1.};
      Eigen::Array<double, 6, 7> repulsion_damping_mask = Eigen::MatrixXd::Zero(6, 7);
      
      Eigen::Array<double, 6, 7> repulsion_damping_multipliers;
      // Eigen::Matrix<double, 3, 7> repulsion_translation;
      // Eigen::Array<double, 1, 7> repulsion_translation_norms;
      // Eigen::Array<double, 3, 7> repulsion_directions;

      Eigen::Array<double, 6, 1> cart_damping_redirection;
      // double cart_damping_force_scalar;

      // Eigen::Matrix<double,6,7> cart_spring_forces;
      // Eigen::Array<double,6,7> cart_damping_forces;
      // Eigen::Matrix<double,6,1> cart_damping_force;

      

    
    // Eigen::Array<double, 7, 1> damping_constants = -6 * sqrt(spring_constants);                // N 
    
    double obj_rescale = 0.8;
    double timeout_scaling = 1.;


    //////////////////// Subscriber for all forces
    std::vector<int> dimension_2 = {0, 0, 0, 0, 0, 0, 0};
    std::vector<double> irregular_vector;
    std::vector<Eigen::MatrixXd> irregular_matrices;

    Eigen::ArrayXXd matrix = Eigen::ArrayXXd::Zero(6, 49);

    int bookmark = 0;



    // Variables for alternative rep controller
    // Eigen::ArrayXXd distances, Fpot_norms, damping_coeffs, damp_colinear_len, Fdamp_norms, Frep_norms, Frep_all

    double spring_constant = max_EE_repulsion_force/(max_dist-min_dist);
    double damping_coeff = 4. * std::sqrt(spring_constant);
    Eigen::Array<double, 6, 1> Frep = Eigen::ArrayXXd::Zero(6, 1);
    double nonlin_stiffness = 4.;

    int dim = 20;
    Eigen::ArrayXXd distances = Eigen::ArrayXXd::Zero(6, 49);
    Eigen::ArrayXXd Fpot_norms = Eigen::ArrayXXd::Zero(1, 49);
    Eigen::ArrayXXd Frep_all = Eigen::ArrayXXd::Zero(6, 49);
    Eigen::ArrayXXd damping_coeffs = Eigen::ArrayXXd::Zero(1, 49);
    Eigen::ArrayXXd damp_colinear_len = Eigen::ArrayXXd::Zero(1, 49);
    Eigen::ArrayXXd Fdamp_norms = Eigen::ArrayXXd::Zero(1, 49);
    Eigen::ArrayXXd Frep_norms = Eigen::ArrayXXd::Zero(1, 49);

    Eigen::ArrayXXd repulsion_translation = Eigen::ArrayXXd::Zero(3, 49);
    Eigen::ArrayXXd repulsion_translation_norms = Eigen::ArrayXXd::Zero(1, 49);
    Eigen::ArrayXXd repulsion_directions = Eigen::ArrayXXd::Zero(3, 49);

    Eigen::ArrayXXd norm_mask = Eigen::ArrayXXd::Zero(1, 49);

    Eigen::ArrayXXd Fpot_rot_norms = Eigen::ArrayXXd::Zero(1, 49);
    Eigen::ArrayXXd rot_damping_coeffs = Eigen::ArrayXXd::Zero(1, 49);
    Eigen::ArrayXXd rot_damp_colinear_len = Eigen::ArrayXXd::Zero(1, 49);
    Eigen::ArrayXXd Fdamp_rot_norms = Eigen::ArrayXXd::Zero(1, 49);
    Eigen::ArrayXXd Frep_rot_norms = Eigen::ArrayXXd::Zero(1, 49);

    Eigen::ArrayXXd repulsion_rotation = Eigen::ArrayXXd::Zero(3, 49);
    Eigen::ArrayXXd repulsion_rotation_norms = Eigen::ArrayXXd::Zero(1, 49);
    Eigen::ArrayXXd repulsion_rot_directions = Eigen::ArrayXXd::Zero(3, 49);
    ////////////////////

    //////////////////// Variables for offloaded spring force and damping calculation

    Eigen::Array<double, 6, 7> repulsion_Ipot;
    Eigen::Array<double, 6, 7> repulsion_Damper;

    Eigen::Array<double, 3, 7> damping_trans;
    Eigen::Array<double, 3, 7> damping_rot;

    Eigen::Array<double, 3, 7> damping_directions_trans;
    Eigen::Array<double, 3, 7> damping_directions_rot;

    Eigen::Array<double, 3, 1> cart_spd_trans;
    Eigen::Array<double, 3, 1> cart_spd_rot;

    Eigen::Array<double, 3, 1> cart_damping_direction_trans;
    Eigen::Array<double, 3, 1> cart_damping_force;
    double cart_damping_force_scalar;

    Eigen::Array<double, 3, 1> cart_damping_direction_rot;
    Eigen::Array<double, 3, 1> cart_damping_moment;
    double cart_damping_moment_scalar;

    Eigen::Array<double, 6, 1> repulsion_Idamp;
    



    // Error logging
    int csv_counter = 0;
    
    const char* path1 = "/home/tom/Documents/Presentation/Frep_EE.csv";
    const char* path2 = "/home/tom/Documents/Presentation/tau_rep.csv";
    const char* path3 = "/home/tom/Documents/Presentation/Fdamp_norms.csv";

    std::ofstream outFile1;
    std::ofstream outFile2;
    std::ofstream outFile3;   

    //Logging
    int outcounter = 0;
    const int update_frequency = 2; //frequency for update outputs

    //Integrator
    Eigen::Matrix<double, 6, 1> I_error = Eigen::MatrixXd::Zero(6, 1);                      // pose error (6d)
    Eigen::Matrix<double, 6, 1> I_F_error = Eigen::MatrixXd::Zero(6, 1);                    // force error integral
    Eigen::Matrix<double, 6, 1> integrator_weights = 
      (Eigen::MatrixXd(6,1) << 75.0, 75.0, 75.0, 75.0, 75.0, 4.0).finished();
    Eigen::Matrix<double, 6, 1> max_I = 
      (Eigen::MatrixXd(6,1) << 30.0, 30.0, 30.0, 50.0, 50.0, 2.0).finished();

   
  
    std::mutex position_and_orientation_d_target_mutex_;

    //Flags
    bool config_control = false;           // sets if we want to control the configuration of the robot in nullspace
    bool do_logging = false;               // set if we do log values

    //Filter-parameters
    double filter_params_{0.001};
    int mode_ = 3;

    // Timer Flags
    double repulsion_date;
};
}  // namespace cartesian_impedance_control
