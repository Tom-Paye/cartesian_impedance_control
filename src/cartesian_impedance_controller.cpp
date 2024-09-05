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

#include <cartesian_impedance_control/cartesian_impedance_controller.hpp>

#include <cassert>
#include <cmath>
#include <exception>
#include <string>

#include <Eigen/Eigen>

namespace {

template <class T, size_t N>
std::ostream& operator<<(std::ostream& ostream, const std::array<T, N>& array) {
  ostream << "[";
  std::copy(array.cbegin(), array.cend() - 1, std::ostream_iterator<T>(ostream, ","));
  std::copy(array.cend() - 1, array.cend(), std::ostream_iterator<T>(ostream));
  ostream << "]";
  return ostream;
}
}

namespace cartesian_impedance_control {

void CartesianImpedanceController::update_stiffness_and_references(){
  //target by filtering
  /** at the moment we do not use dynamic reconfigure and control the robot via D, K and T **/
  //K = filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * K;
  //D = filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * D;
  nullspace_stiffness_ = filter_params_ * nullspace_stiffness_target_ + (1.0 - filter_params_) * nullspace_stiffness_;
  //std::lock_guard<std::mutex> position_d_target_mutex_lock(position_and_orientation_d_target_mutex_);
  position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
  orientation_d_ = orientation_d_.slerp(filter_params_, orientation_d_target_);
  F_contact_des = 0.05 * F_contact_target + 0.95 * F_contact_des;
}


void CartesianImpedanceController::arrayToMatrix(const std::array<double,7>& inputArray, Eigen::Matrix<double,7,1>& resultMatrix)
{
 for(long unsigned int i = 0; i < 7; ++i){
     resultMatrix(i,0) = inputArray[i];
   }
}

void CartesianImpedanceController::arrayToMatrix(const std::array<double,6>& inputArray, Eigen::Matrix<double,6,1>& resultMatrix)
{
 for(long unsigned int i = 0; i < 6; ++i){
     resultMatrix(i,0) = inputArray[i];
   }
}

Eigen::Matrix<double, 7, 1> CartesianImpedanceController::saturateTorqueRate(
  const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
  const Eigen::Matrix<double, 7, 1>& tau_J_d_M) {  
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (u_int i = 0; i < 7; i++) {
  double difference = tau_d_calculated[i] - tau_J_d_M[i];
  tau_d_saturated[i] =
         tau_J_d_M[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}


inline void pseudoInverse(const Eigen::MatrixXd& M_, Eigen::MatrixXd& M_pinv_, bool damped = true) {
  double lambda_ = damped ? 0.2 : 0.0;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(M_, Eigen::ComputeFullU | Eigen::ComputeFullV);   
  Eigen::JacobiSVD<Eigen::MatrixXd>::SingularValuesType sing_vals_ = svd.singularValues();
  Eigen::MatrixXd S_ = M_;  // copying the dimensions of M_, its content is not needed.
  S_.setZero();

  for (int i = 0; i < sing_vals_.size(); i++)
     S_(i, i) = (sing_vals_(i)) / (sing_vals_(i) * sing_vals_(i) + lambda_ * lambda_);

  M_pinv_ = Eigen::MatrixXd(svd.matrixV() * S_.transpose() * svd.matrixU().transpose());
}


controller_interface::InterfaceConfiguration
CartesianImpedanceController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (int i = 1; i <= num_joints; ++i) {
    config.names.push_back(robot_name_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}


controller_interface::InterfaceConfiguration CartesianImpedanceController::state_interface_configuration()
  const {
  controller_interface::InterfaceConfiguration state_interfaces_config;
  state_interfaces_config.type = controller_interface::interface_configuration_type::INDIVIDUAL;

  for (int i = 1; i <= num_joints; ++i) {
    state_interfaces_config.names.push_back(robot_name_ + "_joint" + std::to_string(i) + "/position");
    state_interfaces_config.names.push_back(robot_name_ + "_joint" + std::to_string(i) + "/velocity");
  }

  for (const auto& franka_robot_model_name : franka_robot_model_->get_state_interface_names()) {
    state_interfaces_config.names.push_back(franka_robot_model_name);
    std::cout << franka_robot_model_name << std::endl;
  }

  const std::string full_interface_name = robot_name_ + "/" + state_interface_name_;

  return state_interfaces_config;
}


CallbackReturn CartesianImpedanceController::on_init() {
   UserInputServer input_server_obj(&position_d_target_, &rotation_d_target_, &K, &D, &T);
   std::thread input_thread(&UserInputServer::main, input_server_obj, 0, nullptr);
   input_thread.detach();
   return CallbackReturn::SUCCESS;
}


CallbackReturn CartesianImpedanceController::on_configure(const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
  franka_semantic_components::FrankaRobotModel(robot_name_ + "/" + k_robot_model_interface_name,
                                               robot_name_ + "/" + k_robot_state_interface_name));
                                               
  try {
    rclcpp::QoS qos_profile(1); // Depth of the message queue
    qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    franka_state_subscriber = get_node()->create_subscription<franka_msgs::msg::FrankaRobotState>(
    "franka_robot_state_broadcaster/robot_state", qos_profile, 
    std::bind(&CartesianImpedanceController::topic_callback, this, std::placeholders::_1));
    std::cout << "Succesfully subscribed to robot_state_broadcaster" << std::endl;
  }

  catch (const std::exception& e) {
    fprintf(stderr,  "Exception thrown during publisher creation at configure stage with message : %s \n",e.what());
    return CallbackReturn::ERROR;
    }


  RCLCPP_DEBUG(get_node()->get_logger(), "configured successfully");

  try {
    // rclcpp::QoS qos_profile(10); // Depth of the message queue
    // qos_profile.reliability(RMW_QOS_POLICY_RELIABILITY_RELIABLE);
    // repulsion_subscriber = get_node()->create_subscription<messages_fr3::msg::Array2d>(
    // "repulsion_forces", 10, 
    // std::bind(&CartesianImpedanceController::repulsion_topic_callback, this, std::placeholders::_1));
    repulsion_subscriber = get_node()->create_subscription<messages_fr3::msg::IrregularDistArray>(
    "repulsion_forces", 10, 
    std::bind(&CartesianImpedanceController::repulsion_topic_callback, this, std::placeholders::_1));
    std::cout << "Succesfully subscribed to repulsion_force_broadcaster" << std::endl;
  }

  catch (const std::exception& e) {
    fprintf(stderr,  "Exception thrown during publisher creation at configure stage with message : %s \n",e.what());
    return CallbackReturn::ERROR;
    }


  RCLCPP_DEBUG(get_node()->get_logger(), "configured successfully");
  
  return CallbackReturn::SUCCESS;
}


CallbackReturn CartesianImpedanceController::on_activate(
  const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);

  std::array<double, 16> initial_pose = franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_pose.data()));
  position_d_ = initial_transform.translation();
  orientation_d_ = Eigen::Quaterniond(initial_transform.rotation());
  std::cout << "Completed Activation process" << std::endl;
  return CallbackReturn::SUCCESS;
}


controller_interface::CallbackReturn CartesianImpedanceController::on_deactivate(
  const rclcpp_lifecycle::State& /*previous_state*/) {
  franka_robot_model_->release_interfaces();
  return CallbackReturn::SUCCESS;
}

// std::array<double, 6> CartesianImpedanceController::convertToStdArray(geometry_msgs::msg::WrenchStamped& wrench) {
//     std::array<double, 6> result;
//     result[0] = wrench.wrench.force.x;
//     result[1] = wrench.wrench.force.y;
//     result[2] = wrench.wrench.force.z;
//     result[3] = wrench.wrench.torque.x;
//     result[4] = wrench.wrench.torque.y;
//     result[5] = wrench.wrench.torque.z;
//     return result;
// }

void CartesianImpedanceController::topic_callback(const std::shared_ptr<franka_msgs::msg::FrankaRobotState> msg) {
  std::array< double, 6 > O_F_ext_hat_K = msg->o_f_ext_hat_k;
  // O_F_ext_hat_K = convertToStdArray(&wrench);
  arrayToMatrix(O_F_ext_hat_K, O_F_ext_hat_K_M);
}

// void CartesianImpedanceController::repulsion_topic_callback(const std::shared_ptr<messages_fr3::msg::Array2d> msg) {
//   int width = msg->width;
//   int height = msg->height;
//   std::vector<double> array = msg->array;
//   // for (int idx=0; idx<42; idx++) {
//   //   array[idx] = msg->array[idx];
//   // }
//   if (width != 7 || height != 6) {
//     std::cout << "Error: repulsion_forces message has the wrong dimensions" << std::endl;
//     std::cout << "Dimensions are:"<< height << " by " << width << std::endl;
//     std::vector<double> array[6][7] = {};
//   }

//   // std::cout << "Dimensions are:"<< height << " by " << width << std::endl;
//   // Eigen::ArrayXXd rel_forces(6, 7);
//   // Eigen::Map<Eigen::Array<double, 6, 7>> rel_forces(array.data(), height, width);
//   // normalized_rep_to_rep_forces(rel_forces);

//   Eigen::Map<Eigen::Matrix<double, 6, 7>> forces(array.data(), height, width);
//   // Eigen::Matrix<double, 6, 7> forces = rel_forces.matrix();
  
//   repulsive_dists = forces;
//   // std::cout << "repulsive_dists received [N]" << std::endl;
//   // std::cout << repulsive_dists << std::endl;

//   repulsion_date = get_node()->get_clock()->now().nanoseconds();

// }

void CartesianImpedanceController::repulsion_topic_callback(const std::shared_ptr<messages_fr3::msg::IrregularDistArray> msg) {
  dimension_2 = msg->dimension2;
  irregular_vector = msg->array;
  // irregular_matrices.clear();
  // for (int idx=0; idx<42; idx++) {
  //   array[idx] = msg->array[idx];
  // }
  bookmark = 0;
  // for (int i=0; i<7; i++){
  //   int dim = dimension_2[i];
  //   matrix.resize(6, dim);
  //       for (int i = 0; i < 6; ++i) {
  //           for (int j = 0; j < dim; ++j) {
  //               matrix(i, j) = irregular_vector[bookmark+i*dim+j];
  //           }
  //       }

    
  //   // Eigen::MatrixXd matrix= irregular_vector.[slice(bookmark, bookmark + dim * 6, 1)];
  //   // Eigen::Matrix<double, 6, dim> matrix= irregular_vector.pop_back(dim * 6);
  //   irregular_matrices.push_back(matrix);
  //   bookmark += 6*dim;
  // }

  // alternatively, kep the data as a vector and store only the index ranges we need, assigning them only at 

  repulsion_date = get_node()->get_clock()->now().nanoseconds();

}

// void CartesianImpedanceController::normalized_rep_to_rep_forces(Eigen::Array<double, 6, 7> relative_forces) {

//   /*
//   Takes a list of pseudo-forces applied to every joint on the robot and 
//   rescales it to have forces in Newtons and moments in Nm

//   Pseudo-forces are the result of processing minimum link distances into an interaction,
//   then resizing this interaction so it is only expressed onto the actionnable joints
//   */

//   ///////////////Rescaling///////////
//   // Expect the first column to correspond to base link (attached to the ground, no DOF)
//   // Expect any link after link 7 to be hands / fingers
  
//   // Eigen::ArrayXXd max_moments;
//   // max_moments<<87., 87., 87., 87., 12., 12., 12.;



//   // for (int i=0;i<6;i++){
//   //   moments_array.row(i) += max_moments;
//   //   //this relates all spring forces and moments by a factor of 1m
//   //   // as in Nm / m = N, but a spring constant is in N/m, so we just say the force is just this
//   //   // spring moment constant applied 1m away 
//   // }
  
//   // relative_forces = relative_forces * moments_array * 2;




// }

void CartesianImpedanceController::updateJointStates() {
  for (auto i = 0; i < num_joints; ++i) {
    const auto& position_interface = state_interfaces_.at(2 * i);
    const auto& velocity_interface = state_interfaces_.at(2 * i + 1);
    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");
    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

 
// void CartesianImpedanceController::calcRepulsiveTorque(Eigen::Matrix<double, 6, 7> repulsive_dists) {
//   // PLAN: for each force in the vector, apply the jacobian of the joint to get the torque inputs
//   // std::cout << "repulsive_dists [N]" << std::endl;
//   // std::cout << repulsive_dists << std::endl;
  
//   double time_now = get_node()->get_clock()->now().nanoseconds();
//   // std::cout << "time" << std::endl;
//   // std::cout << time_now - repulsion_date << std::endl;
//   double d_00_t = (time_now - repulsion_date) *0.000000001;

//  // typical cam frequency is 22 hz ==> delay is 1/20 = 0.05s
//   if (repulsive_dists.isZero(0) || d_00_t > 0.05) {
//     // tau_repulsion = tau_repulsion * 0.9997;

//     timeout_scaling = timeout_scaling * 0.9997;
//     // repulsion_damping_multipliers = damping_constants.array().topRows();

//     tau_damping = tau_damping * 0;
//     for (int i=0; i<num_joints; ++i) {
//       std::array<double, 42> jacobian_array_i =  franka_robot_model_->getZeroJacobian(franka::Frame(i));
//       Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian_i(jacobian_array_i.data());

//       // Eigen::Array<double, 6, 1> mask_col = mask.col(i);
//       Eigen::Array<double, 6, 1> unit_cart_spd = jacobian_i * dq_;  //unit_jt_spd
//       // tau_damping_i = jacobian_i.transpose() * Sm * (mask_col * unit_cart_spd.array()).matrix();
//       cart_damping_force_scalar = (repulsion_damping_multipliers.col(i) * unit_cart_spd.array()).sum();
//       cart_damping_redirection = repulsion_damping_mask.col(i) * -1 * std::signbit(cart_damping_force_scalar);
//       cart_damping_force = cart_damping_force_scalar * cart_damping_redirection;
//       tau_damping_i = jacobian_i.transpose() * Sm * cart_damping_force.matrix();
      
//       tau_damping_i = jacobian_i.transpose() * Sm * cart_damping_force;
//       cart_damping_forces.col(i) = cart_damping_force;
      
//       tau_damping += tau_damping_i;
//     }

//     return;
//   }
  
//   // F_impedance = -1 * (D * (jacobian * dq_) + K * error /*+ I_error*/);
//   if (d_00_t > 0.006) {
//     // std::cout << "d_00_t > 0.01" << std::endl;
//     //// tau_repulsion = (tau_spring.array() * spring_constants).matrix();
//     // tau_spring = tau_spring * 0.099999;
//     tau_damping = tau_damping * 0;
//     for (int i=0; i<num_joints; ++i) {
//       std::array<double, 42> jacobian_array_i =  franka_robot_model_->getZeroJacobian(franka::Frame(i));
//       Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian_i(jacobian_array_i.data());

//       // Eigen::Array<double, 6, 1> mask_col = mask.col(i);
//       Eigen::Array<double, 6, 1> unit_cart_spd = jacobian_i * dq_;  //unit_jt_spd
//       // tau_damping_i = jacobian_i.transpose() * Sm * (mask_col * unit_cart_spd.array()).matrix();
//       cart_damping_force_scalar = (repulsion_damping_multipliers.col(i) * unit_cart_spd.array()).sum();
//       cart_damping_redirection = repulsion_damping_mask.col(i) * -1 * std::signbit(cart_damping_force_scalar);
//       cart_damping_force = cart_damping_force_scalar * cart_damping_redirection;
//       tau_damping_i = jacobian_i.transpose() * Sm * cart_damping_force.matrix();

//       tau_damping_i = jacobian_i.transpose() * Sm * cart_damping_force;
//       cart_damping_forces.col(i) = cart_damping_force;

//       tau_damping += tau_damping_i;
//     }
//   }

  
//   if (d_00_t <= 0.006) {
//     // std::cout << "d_00_t <= 0.01" << std::endl;

//     // mask = (repulsive_dists.array().abs()>1e-10).cast<double>();       // every col is a joint, with a 1 in every row where here is a rep force

//     //////////////////////// Calculate damper in cartesian space along the direction of the spring
//     repulsion_translation = repulsive_dists.topRows(3);

//     repulsion_translation_norms = repulsion_translation.colwise().norm();

//     repulsion_directions = (repulsion_translation.array().rowwise() /= repulsion_translation_norms);
//     repulsion_directions = repulsion_directions.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });

//     repulsion_damping_mask.topRows(3) = repulsion_directions;

//     repulsion_damping_multipliers = (repulsion_damping_mask.matrix() * damping_constants).array();
//     // This determines the orientation and strength of the damper asociated with each spring force

//     ////////////////////////


//     timeout_scaling = 1.;
//     tau_spring = tau_spring * 0.;
//     tau_damping = tau_damping * 0.;

//     cart_spring_forces = repulsive_dists * spring_constants;

//     for (int i=0; i<num_joints; ++i) {

//       std::array<double, 42> jacobian_array_i =  franka_robot_model_->getZeroJacobian(franka::Frame(i));
//       Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian_i(jacobian_array_i.data());

      
//       tau_spring_i = jacobian_i.transpose() * Sm * (cart_spring_forces.col(i));
//       // tau_spring_i = jacobian_i.transpose() * Sm * (repulsive_dists.col(i));

//       /////////////////// Nonlinear APF
//       Fpot = nonlin_stiffness * (1/D - 1/Q) ** 0.6

//       ///////////////////
//       tau_spring += tau_spring_i;

//       // Eigen::Array<double, 6, 1> mask_col = mask.col(i);

//       Eigen::Array<double, 6, 1> unit_cart_spd = jacobian_i * dq_;  //unit_jt_spd

//       // tau_damping_i = jacobian_i.transpose() * Sm * (mask_col * unit_cart_spd.array()).matrix();
//       cart_damping_force_scalar = (repulsion_damping_multipliers.col(i) * unit_cart_spd.array()).sum();
//       cart_damping_redirection = repulsion_damping_mask.col(i) * -1 * std::signbit(cart_damping_force_scalar);
//       cart_damping_force = cart_damping_force_scalar * cart_damping_redirection;
//       tau_damping_i = jacobian_i.transpose() * Sm * cart_damping_force.matrix();

//       cart_damping_forces.col(i) = cart_damping_force;


//       tau_damping += tau_damping_i;
//     }
//     // tau_repulsion = (tau_spring.array() * spring_constants).matrix();
//   }

//   // TODO: change these so K and D are applied before the jacobians, as intended
  
//   // Eigen::Array<double, 7, 1> p_sign_mask = (tau_repulsion.array()>0).cast<double>();
//   // Eigen::Array<double, 7, 1> n_sign_mask = (tau_repulsion.array()<0).cast<double>();
//   // Eigen::Array<double, 7, 1> pos_mask = tau_repulsion.array().min(max_moments);
//   // Eigen::Array<double, 7, 1> neg_mask = tau_repulsion.array().max(-max_moments);
//   // tau_repulsion << (p_sign_mask * pos_mask + n_sign_mask * neg_mask);
  
//   // tau_damping = (tau_damping.array() * damping_constants).matrix();

//   // std::cout << "tau_spring [Nm]" << std::endl;
//   // std::cout << tau_repulsion << std::endl;
//   // std::cout << "tau_damping_part [Nm]" << std::endl;
//   // std::cout << tau_damping << std::endl;
//   // std::cout << "tau_repulsion raw [Nm]" << std::endl;
//   // std::cout << tau_repulsion << std::endl;
//   tau_repulsion = (tau_spring.array()*timeout_scaling + tau_damping.array()*std::sqrt(timeout_scaling)).matrix();
//   // repulsive_dists = repulsive_dists * 0.99;


  


// }

void CartesianImpedanceController::calcRepulsiveTorque(std::vector<double> irregular_vector) {
  // PLAN: for each force in the vector, apply the jacobian of the joint to get the torque inputs
  // std::cout << "repulsive_dists [N]" << std::endl;
  // std::cout << repulsive_dists << std::endl;
  
  double time_now = get_node()->get_clock()->now().nanoseconds();
  // std::cout << "time" << std::endl;
  // std::cout << time_now - repulsion_date << std::endl;
  double d_00_t = (time_now - repulsion_date) *0.000000001;

//  // typical cam frequency is 22 hz ==> delay is 1/20 = 0.05s
//   if (repulsive_dists.isZero(0) || d_00_t > 0.05) {
//     // tau_repulsion = tau_repulsion * 0.9997;

//     timeout_scaling = timeout_scaling * 0.9997;
//     // repulsion_damping_multipliers = damping_constants.array().topRows();

//     tau_damping = tau_damping * 0;
//     for (int i=0; i<num_joints; ++i) {
//       std::array<double, 42> jacobian_array_i =  franka_robot_model_->getZeroJacobian(franka::Frame(i));
//       Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian_i(jacobian_array_i.data());

//       // Eigen::Array<double, 6, 1> mask_col = mask.col(i);
//       Eigen::Array<double, 6, 1> unit_cart_spd = jacobian_i * dq_;  //unit_jt_spd
//       // tau_damping_i = jacobian_i.transpose() * Sm * (mask_col * unit_cart_spd.array()).matrix();
//       cart_damping_force_scalar = (repulsion_damping_multipliers.col(i) * unit_cart_spd.array()).sum();
//       cart_damping_redirection = repulsion_damping_mask.col(i) * -1 * std::signbit(cart_damping_force_scalar);
//       cart_damping_force = cart_damping_force_scalar * cart_damping_redirection;
//       tau_damping_i = jacobian_i.transpose() * Sm * cart_damping_force.matrix();
      
//       tau_damping_i = jacobian_i.transpose() * Sm * cart_damping_force;
//       cart_damping_forces.col(i) = cart_damping_force;
      
//       tau_damping += tau_damping_i;
//     }

//     return;
//   }
  
//   // F_impedance = -1 * (D * (jacobian * dq_) + K * error /*+ I_error*/);
//   if (d_00_t > 0.006) {
//     // std::cout << "d_00_t > 0.01" << std::endl;
//     //// tau_repulsion = (tau_spring.array() * spring_constants).matrix();
//     // tau_spring = tau_spring * 0.099999;
//     tau_damping = tau_damping * 0;
//     for (int i=0; i<num_joints; ++i) {
//       std::array<double, 42> jacobian_array_i =  franka_robot_model_->getZeroJacobian(franka::Frame(i));
//       Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian_i(jacobian_array_i.data());

//       // Eigen::Array<double, 6, 1> mask_col = mask.col(i);
//       Eigen::Array<double, 6, 1> unit_cart_spd = jacobian_i * dq_;  //unit_jt_spd
//       // tau_damping_i = jacobian_i.transpose() * Sm * (mask_col * unit_cart_spd.array()).matrix();
//       cart_damping_force_scalar = (repulsion_damping_multipliers.col(i) * unit_cart_spd.array()).sum();
//       cart_damping_redirection = repulsion_damping_mask.col(i) * -1 * std::signbit(cart_damping_force_scalar);
//       cart_damping_force = cart_damping_force_scalar * cart_damping_redirection;
//       tau_damping_i = jacobian_i.transpose() * Sm * cart_damping_force.matrix();

//       tau_damping_i = jacobian_i.transpose() * Sm * cart_damping_force;
//       cart_damping_forces.col(i) = cart_damping_force;

//       tau_damping += tau_damping_i;
//     }
//   }
  if (d_00_t > 0.004) {}
  
  if (d_00_t <= 0.004) {
  


    // timeout_scaling = 1.;
    // tau_spring = tau_spring * 0.;
    // tau_damping = tau_damping * 0.;
    tau_repulsion = tau_repulsion * 0.;

  


    for (int i=0; i<num_joints; ++i) {

      // Initialize the size-varying matrices

      dim = dimension_2[i];

      if (dim>0) {
        Eigen::ArrayXXd distances = Eigen::ArrayXXd::Zero(6, dim);

        Eigen::ArrayXXd Fpot_norms = Eigen::ArrayXXd::Zero(1, dim);
        Eigen::ArrayXXd Frep_all = Eigen::ArrayXXd::Zero(6, dim);
        Eigen::ArrayXXd damping_coeffs = Eigen::ArrayXXd::Zero(1, dim);
        Eigen::ArrayXXd damp_colinear_len = Eigen::ArrayXXd::Zero(1, dim);
        Eigen::ArrayXXd Fdamp_norms = Eigen::ArrayXXd::Zero(1, dim);
        Eigen::ArrayXXd Frep_norms = Eigen::ArrayXXd::Zero(1, dim);

        Eigen::ArrayXXd repulsion_translation = Eigen::ArrayXXd::Zero(3, dim);
        Eigen::ArrayXXd repulsion_translation_norms = Eigen::ArrayXXd::Zero(1, dim);
        Eigen::ArrayXXd repulsion_directions = Eigen::ArrayXXd::Zero(3, dim);

        Eigen::ArrayXXd norm_mask = Eigen::ArrayXXd::Zero(1, dim);



        Eigen::ArrayXXd Fpot_rot_norms = Eigen::ArrayXXd::Zero(1, dim);
        Eigen::ArrayXXd rot_damping_coeffs = Eigen::ArrayXXd::Zero(1, dim);
        Eigen::ArrayXXd rot_damp_colinear_len = Eigen::ArrayXXd::Zero(1, dim);
        Eigen::ArrayXXd Fdamp_rot_norms = Eigen::ArrayXXd::Zero(1, dim);
        Eigen::ArrayXXd Frep_rot_norms = Eigen::ArrayXXd::Zero(1, dim);

        Eigen::ArrayXXd repulsion_rotation = Eigen::ArrayXXd::Zero(3, dim);
        Eigen::ArrayXXd repulsion_rotation_norms = Eigen::ArrayXXd::Zero(1, dim);
        Eigen::ArrayXXd repulsion_rot_directions = Eigen::ArrayXXd::Zero(3, dim);

        // int dim = dimension_2[i];
        // Eigen::ArrayXXd distances(6, dim);
        // Eigen::ArrayXXd Fpot_norms(1, dim), damping_coeffs(1, dim), damp_colinear_len(1, dim), Fdamp_norms(1, dim), Frep_norms(1, dim);
        // Eigen::ArrayXXd Frep_all(6, dim);

        // Eigen::ArrayXXd repulsion_translation(3, dim);
        // Eigen::ArrayXXd repulsion_translation_norms(1, dim);
        // Eigen::ArrayXXd repulsion_directions(3, dim);
        
        // int dim = dimension_2[i];
        // distances.resize(6, dim);
        // Fpot_norms.resize(1, dim); damping_coeffs.resize(1, dim); damp_colinear_len.resize(1, dim); Fdamp_norms.resize(1, dim);
        // Frep_norms.resize(1, dim);
        // Frep_all.resize(6, dim);

        // repulsion_translation.resize(3, dim);
        // repulsion_translation_norms.resize(1, dim);
        // repulsion_directions.resize(3, dim);


        // Get relevant robot data
        std::array<double, 42> jacobian_array_i =  franka_robot_model_->getZeroJacobian(franka::Frame(i));
        Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian_i(jacobian_array_i.data());

        Eigen::Array<double, 6, 1> unit_cart_spd = jacobian_i * dq_;




        for (int k = 0; k < 6; ++k) {
            for (int j = 0; j < dim; ++j) {
                distances(k, j) = irregular_vector[bookmark+k*dim+j];
            }
        }
        bookmark += 6*dim;
      
        if (distances.hasNaN()){
          std::cout << "NaNs in published dists!" << std::endl;
          std::cout << "joint " << i << " dists:" << std::endl;
          std::cout << distances << std::endl;
          std::cout << "All dists:" << std::endl;
          for (const auto& element : irregular_vector) {
              std::cout << element << " ";
          }
          std::cout << std::endl;
          
        }

        distances = distances.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
        distances = distances.unaryExpr([](double v) { return std::isinf(v) ? 0.0 : v; });
      
        

        if (dim>1) {

          // distances = irregular_matrices[i];
          
          repulsion_translation = distances(Eigen::seq(0,2), Eigen::all).array();

          repulsion_translation_norms = repulsion_translation.array().colwise().norm();

          repulsion_directions = repulsion_translation.array().colwise().normalized();
          repulsion_directions = repulsion_directions.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
          repulsion_directions = repulsion_directions.unaryExpr([](double v) { return std::isinf(v) ? 0.0 : v; });


          // Get cartesian force norms along the direction body --> robot
          // norm_mask = (repulsion_translation_norms>max_dist).cast<double>() * 1.;
          norm_mask = (repulsion_translation_norms>max_dist || repulsion_translation_norms==0).cast<double>();
          // Fpot_norms = nonlin_stiffness * ((repulsion_translation_norms + norm_mask).inverse() - 1/max_dist).pow(0.6);
          // Fpot_norms = Fpot_norms.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
          // Fpot_norms = Fpot_norms.unaryExpr([](double v) { return std::isinf(v) ? 0.0 : v; });
          Fpot_norms = spring_constant * (max_dist - repulsion_translation_norms) * (1-norm_mask);

          // damping_coeffs = 2 * (Fpot_norms / repulsion_translation_norms).sqrt(); 
          damping_coeffs = 2 * Fpot_norms.sqrt(); 

          damp_colinear_len = (repulsion_directions.colwise() * unit_cart_spd(Eigen::seq(0,2), Eigen::all).array()).sum();
          // Fdamp_norms = damping_coeffs * damp_colinear_len * (damp_colinear_len * damp_colinear_len.abs().inverse()) * -1;
          Fdamp_norms = damping_coeff * damp_colinear_len * (damp_colinear_len * damp_colinear_len.abs().inverse()) * -1;
          Fdamp_norms = Fdamp_norms.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
          Fdamp_norms = Fdamp_norms.unaryExpr([](double v) { return std::isinf(v) ? 0.0 : v; });
          


          Frep_norms = Fpot_norms + Fdamp_norms;
          

          // giving the forces a dimension again in cartesian space
          Frep_all(Eigen::seq(0,2), Eigen::all) = repulsion_directions * Frep_norms;


          // // The same, but for MoF

          repulsion_rotation = distances(Eigen::seq(Eigen::last-2,Eigen::last), Eigen::all).array();

          repulsion_rotation_norms = repulsion_rotation.matrix().colwise().norm().array();

          repulsion_rot_directions = repulsion_rotation.array().colwise().normalized();
          repulsion_rot_directions = repulsion_rot_directions.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
          repulsion_rot_directions = repulsion_rot_directions.unaryExpr([](double v) { return std::isinf(v) ? 0.0 : v; });


          // Get cartesian force norms along the direction body --> robot
          // norm_mask = (repulsion_rotation_norms>max_dist).cast<double>() * 1;
          norm_mask = (repulsion_rotation_norms>max_dist ||repulsion_rotation_norms==0).cast<double>();
          // Fpot_rot_norms = nonlin_stiffness * ((repulsion_rotation_norms+norm_mask).inverse() - 1/max_dist).pow(0.6);
          // Fpot_rot_norms = Fpot_rot_norms.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
          // Fpot_rot_norms = Fpot_rot_norms.unaryExpr([](double v) { return std::isinf(v) ? 0.0 : v; });
          Fpot_norms = spring_constant * (max_dist - repulsion_rotation_norms) * (1-norm_mask);

          // damping_coeffs = 2 * (Fpot_rot_norms / repulsion_rotation_norms).sqrt(); 
          rot_damping_coeffs = 2 * Fpot_rot_norms.sqrt(); 

          rot_damp_colinear_len = (repulsion_rot_directions.colwise() * unit_cart_spd(Eigen::seq(Eigen::last-2,Eigen::last), Eigen::all).array()).sum();
          // Fdamp_rot_norms = rot_damping_coeffs * rot_damp_colinear_len * (rot_damp_colinear_len * rot_damp_colinear_len.abs().inverse()) * -1;
          Fdamp_rot_norms = damping_coeff * rot_damp_colinear_len * (rot_damp_colinear_len * rot_damp_colinear_len.abs().inverse()) * -1;
          Fdamp_rot_norms = Fdamp_rot_norms.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
          Fdamp_rot_norms = Fdamp_rot_norms.unaryExpr([](double v) { return std::isinf(v) ? 0.0 : v; });
          


          Frep_rot_norms = Fpot_rot_norms + Fdamp_rot_norms;
          

          // giving the forces a dimension again in cartesian space
          Frep_all(Eigen::seq(Eigen::last-2,Eigen::last), Eigen::all) = repulsion_rot_directions * Frep_rot_norms;


          Frep = Frep_all.rowwise().sum();

        }

        if (dim==1) {
          repulsion_translation = distances(Eigen::seq(0,2), Eigen::all);

          repulsion_translation_norms = repulsion_translation.matrix().norm();

          repulsion_directions = (repulsion_translation / repulsion_translation_norms);
          repulsion_directions = repulsion_directions.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
          repulsion_directions = repulsion_directions.unaryExpr([](double v) { return std::isinf(v) ? 0.0 : v; });

          // Get cartesian force norms along the direction body --> robot
          // norm_mask = (repulsion_translation_norms>max_dist).cast<double>() * 1;
          norm_mask = (repulsion_translation_norms>max_dist || repulsion_translation_norms==0).cast<double>();
          // Fpot_norms = nonlin_stiffness * ((repulsion_translation_norms).inverse() - 1/max_dist).pow(0.6);
          // Fpot_norms = Fpot_norms.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
          // Fpot_norms = Fpot_norms.unaryExpr([](double v) { return std::isinf(v) ? 0.0 : v; });
          Fpot_norms = spring_constant * (max_dist - repulsion_translation_norms)* (1-norm_mask);
          
          

          // damping_coeffs = 2 * (Fpot_norms / repulsion_translation_norms).sqrt(); 
          damping_coeffs = 2 * Fpot_norms.sqrt(); 

          damp_colinear_len = (repulsion_directions * unit_cart_spd(Eigen::seq(0,2), Eigen::all).array()).sum();
          // Fdamp_norms = damping_coeffs * damp_colinear_len * (damp_colinear_len * damp_colinear_len.abs().inverse()) * -1;
          Fdamp_norms = damping_coeff * damp_colinear_len * (damp_colinear_len * damp_colinear_len.abs().inverse()) * -1;
          Fdamp_norms = Fdamp_norms.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
          Fdamp_norms = Fdamp_norms.unaryExpr([](double v) { return std::isinf(v) ? 0.0 : v; });


          Frep_norms = Fpot_norms + Fdamp_norms;
          

          // giving the forces a dimension again in cartesian space
          Frep_all(Eigen::seq(0,2), Eigen::all) = repulsion_directions * Frep_norms;


          // // Same, but for MoF
          repulsion_rotation = distances(Eigen::seq(Eigen::last-2,Eigen::last), Eigen::all);

          repulsion_rotation_norms = repulsion_rotation.matrix().norm();

          repulsion_rot_directions = (repulsion_rotation / repulsion_rotation_norms);
          repulsion_rot_directions = repulsion_rot_directions.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
          repulsion_rot_directions = repulsion_rot_directions.unaryExpr([](double v) { return std::isinf(v) ? 0.0 : v; });

          // Get cartesian force norms along the direction body --> robot
          // norm_mask = (repulsion_rotation_norms>max_dist).cast<double>() * 1;
          norm_mask = (repulsion_rotation_norms>max_dist || repulsion_rotation_norms==0).cast<double>();
          // Fpot_rot_norms = nonlin_stiffness * ((repulsion_rotation_norms+norm_mask).inverse() - 1/max_dist).pow(0.6);
          // Fpot_rot_norms = Fpot_rot_norms.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
          // Fpot_rot_norms = Fpot_rot_norms.unaryExpr([](double v) { return std::isinf(v) ? 0.0 : v; });
          Fpot_rot_norms = spring_constant * (max_dist - repulsion_rotation_norms) * (1-norm_mask);
          

          // damping_coeffs = 2 * (Fpot_rot_norms / repulsion_rotation_norms).sqrt(); 
          rot_damping_coeffs = 2 * Fpot_rot_norms.sqrt(); 

          rot_damp_colinear_len = (repulsion_rot_directions * unit_cart_spd(Eigen::seq(Eigen::last-2,Eigen::last), Eigen::all).array()).sum();
          // Fdamp_rot_norms = rot_damping_coeffs * rot_damp_colinear_len * (rot_damp_colinear_len * rot_damp_colinear_len.abs().inverse()) * -1;
          Fdamp_rot_norms = damping_coeff * rot_damp_colinear_len * (rot_damp_colinear_len * rot_damp_colinear_len.abs().inverse()) * -1;
          Fdamp_rot_norms = Fdamp_rot_norms.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
          Fdamp_rot_norms = Fdamp_rot_norms.unaryExpr([](double v) { return std::isinf(v) ? 0.0 : v; });


          Frep_rot_norms = Fpot_rot_norms + Fdamp_rot_norms;
          

          // giving the forces a dimension again in cartesian space
          Frep_all(Eigen::seq(Eigen::last-2,Eigen::last), Eigen::all) = repulsion_rot_directions * Frep_rot_norms;


          Frep = Frep_all;

        }

        // TODO: split spring and damper forces, save springs and damper multipliers for later use
        Frep = Frep.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
        Frep = Frep.unaryExpr([](double v) { return std::isinf(v) ? 0.0 : v; });
        
        tau_spring_i = jacobian_i.transpose() * Sm * Frep.matrix();




        Frep = Frep.unaryExpr([](double v) { return std::isnan(v) ? 0.0 : v; });
        Frep = Frep.unaryExpr([](double v) { return std::isinf(v) ? 0.0 : v; });
        
        tau_spring_i = jacobian_i.transpose() * Sm * Frep.matrix();

        tau_repulsion += tau_spring_i;

        if (tau_spring_i.hasNaN()){

          std::cout << "Frep" << std::endl;
          std::cout << Frep << std::endl;
          std::cout << "Fpot_rot_norms" << std::endl;
          std::cout << Fpot_rot_norms << std::endl;
          std::cout << "Fdamp_rot_norms" << std::endl;
          std::cout << Fdamp_rot_norms << std::endl;
          std::cout << "repulsion_directions" << std::endl;
          std::cout << repulsion_directions << std::endl;
          std::cout << "repulsion_translation_norms" << std::endl;
          std::cout << repulsion_translation_norms << std::endl;
          std::cout << "repulsion_translation" << std::endl;
          std::cout << repulsion_translation << std::endl;
          std::cout << "Frep_all" << std::endl;
          std::cout << Frep_all << std::endl;
          
        }

        

      }



      

    }
  }

  //   // //logging
  //   //   cart_spring_forces = repulsion_directions.rowwise() * Fpot_norms;
  //   //   cart_damping_forces = repulsion_directions.rowwise() * Fdamp_norms;

    













  // //     // std::array<double, 42> jacobian_array_i =  franka_robot_model_->getZeroJacobian(franka::Frame(i));
  // //     // Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian_i(jacobian_array_i.data());

      
  // //     tau_spring_i = jacobian_i.transpose() * Sm * (cart_spring_forces.col(i));
  // //     // tau_spring_i = jacobian_i.transpose() * Sm * (repulsive_dists.col(i));

  // //     /////////////////// Nonlinear APF
  // //     Fpot = nonlin_stiffness * (1/D - 1/Q) ** 0.6

  // //     ///////////////////
  // //     tau_spring += tau_spring_i;

  // //     // Eigen::Array<double, 6, 1> mask_col = mask.col(i);

  // //     Eigen::Array<double, 6, 1> unit_cart_spd = jacobian_i * dq_;  //unit_jt_spd

  // //     // tau_damping_i = jacobian_i.transpose() * Sm * (mask_col * unit_cart_spd.array()).matrix();
  // //     cart_damping_force_scalar = (repulsion_damping_multipliers.col(i) * unit_cart_spd.array()).sum();
  // //     cart_damping_redirection = repulsion_damping_mask.col(i) * -1 * std::signbit(cart_damping_force_scalar);
  // //     cart_damping_force = cart_damping_force_scalar * cart_damping_redirection;
  // //     tau_damping_i = jacobian_i.transpose() * Sm * cart_damping_force.matrix();

  // //     cart_damping_forces.col(i) = cart_damping_force;


  // //     tau_damping += tau_damping_i;
  // //   }
  // //   // tau_repulsion = (tau_spring.array() * spring_constants).matrix();
  

 
  // // tau_repulsion = (tau_spring.array()*timeout_scaling + tau_damping.array()*std::sqrt(timeout_scaling)).matrix();
  // // // repulsive_dists = repulsive_dists * 0.99;

}

controller_interface::return_type CartesianImpedanceController::update(const rclcpp::Time& /*time*/, const rclcpp::Duration& /*period*/) {  
  // if (outcounter == 0){
  // std::cout << "Enter 1 if you want to track a desired position or 2 if you want to use free floating with optionally shaped inertia" << std::endl;
  // std::cin >> mode_;
  // std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
  // std::cout << "Mode selected" << std::endl;
  // while (mode_ != 1 && mode_ != 2){
  //   std::cout << "Invalid mode, try again" << std::endl;
  //   std::cin >> mode_;
  // }
  // }
  std::array<double, 49> mass = franka_robot_model_->getMassMatrix();
  std::array<double, 7> coriolis_array = franka_robot_model_->getCoriolisForceVector();
  std::array<double, 42> jacobian_array =  franka_robot_model_->getZeroJacobian(franka::Frame::kEndEffector);
  std::array<double, 16> pose = franka_robot_model_->getPoseMatrix(franka::Frame::kEndEffector);
  Eigen::Map<Eigen::Matrix<double, 7, 1>> coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7>> jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 7>> M(mass.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(pose.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.rotation());
  orientation_d_target_ = Eigen::AngleAxisd(rotation_d_target_[0], Eigen::Vector3d::UnitX())
                        * Eigen::AngleAxisd(rotation_d_target_[1], Eigen::Vector3d::UnitY())
                        * Eigen::AngleAxisd(rotation_d_target_[2], Eigen::Vector3d::UnitZ());
  updateJointStates(); 

  
  error.head(3) << position - position_d_;

  if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
    orientation.coeffs() << -orientation.coeffs();
  }
  Eigen::Quaterniond error_quaternion(orientation.inverse() * orientation_d_);
  error.tail(3) << error_quaternion.x(), error_quaternion.y(), error_quaternion.z();
  error.tail(3) << -transform.rotation() * error.tail(3);
  I_error += Sm * dt * integrator_weights.cwiseProduct(error);
  for (int i = 0; i < 6; i++){
    I_error(i,0) = std::min(std::max(-max_I(i,0),  I_error(i,0)), max_I(i,0)); 
  }

  Lambda = (jacobian * M.inverse() * jacobian.transpose()).inverse();
  // Theta = T*Lambda;
  // F_impedance = -1*(Lambda * Theta.inverse() - IDENTITY) * F_ext;
  //Inertia of the robot
  switch (mode_)
  {
  case 1:
    Theta = Lambda;
    F_impedance = -1 * (D * (jacobian * dq_) + K * error /*+ I_error*/);
    break;

  case 2:
    Theta = T*Lambda;
    F_impedance = -1*(Lambda * Theta.inverse() - IDENTITY) * F_ext;
    break;
  case 3: {
    Theta = Lambda;
    Eigen::Array<double, 6, 1> p_sign_mask = (error.head(3).array()>0).cast<double>();
    Eigen::Array<double, 6, 1> n_sign_mask = (error.head(3).array()<0).cast<double>();
    Eigen::Array<double, 6, 1> pos_mask = error.head(3).array().min(impedance_limit_dist);
    Eigen::Array<double, 6, 1> neg_mask = error.head(3).array().max(-impedance_limit_dist);
    error.head(3) << (p_sign_mask * pos_mask + n_sign_mask * neg_mask);

    F_impedance = -1 * (D * std::sqrt(obj_rescale) * (jacobian * dq_) + K * obj_rescale * error);
  }
  
  default:
    break;
  }

  F_ext = 0.9 * F_ext + 0.1 * O_F_ext_hat_K_M; //Filtering 
  I_F_error += dt * Sf* (F_contact_des - F_ext);
  F_cmd = Sf*(0.4 * (F_contact_des - F_ext) + 0.9 * I_F_error + 0.9 * F_contact_des);

  Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7), tau_impedance(7);
  pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

  // calcRepulsiveTorque(repulsive_dists);
  calcRepulsiveTorque(irregular_vector);
  tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
                    jacobian.transpose() * jacobian_transpose_pinv) *
                    (nullspace_stiffness_ * config_control * (q_d_nullspace_ - q_) - //if config_control = true we control the whole robot configuration
                    (2.0 * sqrt(nullspace_stiffness_)) * dq_);  // if config control ) false we don't care about the joint position

  tau_impedance = jacobian.transpose() * Sm * (F_impedance /*+ F_repulsion + F_potential*/) + jacobian.transpose() * Sf * F_cmd;
  // tau_impedance << saturateTorqueRate(tau_impedance, tau_J_d_M * 0.85);  // Saturate impedance torque to prioritize repulsion
  auto tau_d_placeholder = tau_repulsion + tau_impedance + tau_nullspace + coriolis; //add nullspace and coriolis components to desired torque
  tau_d << tau_d_placeholder;
  tau_d << saturateTorqueRate(tau_d, tau_J_d_M);  // Saturate torque rate to avoid discontinuities
  tau_J_d_M = tau_d;

  for (size_t i = 0; i < 7; ++i) {
    command_interfaces_[i].set_value(tau_d(i));
  }
  
  if (outcounter % 1000/update_frequency == 0){
    // std::cout << "F_ext_robot [N]" << std::endl;
    // std::cout << O_F_ext_hat_K << std::endl;
    // std::cout << O_F_ext_hat_K_M << std::endl;
    // std::cout << "Lambda  Thetha.inv(): " << std::endl;
    // std::cout << Lambda*Theta.inverse() << std::endl;
    // std::cout << "tau_d" << std::endl;
    // std::cout << tau_d << std::endl;
    // std::cout << "--------" << std::endl;
    // std::cout << "tau_nullspace" << std::endl;
    // std::cout << tau_nullspace << std::endl;
    // std::cout << "--------" << std::endl;
    // std::cout << "tau_impedance" << std::endl;
    // std::cout << tau_impedance << std::endl;
    // std::cout << "--------" << std::endl;
    // std::cout << "coriolis" << std::endl;
    // std::cout << coriolis << std::endl;
    // std::cout << "--------" << std::endl;
    std::cout << "tau_repulsion" << std::endl;
    std::cout << tau_repulsion << std::endl;
    // std::cout << "Inertia scaling [m]: " << std::endl;
    // std::cout << T << std::endl;

    
    // std::cout << "mask" << std::endl;
    // std::cout << mask << std::endl;
    // std::cout << "tau_spring" << std::endl;
    // std::cout << tau_spring << std::endl;
    // std::cout << "tau_damping" << std::endl;
    // std::cout << tau_damping << std::endl;

    // std::cout << "repulsive_dists" << std::endl;
    // std::cout << repulsive_dists << std::endl;

    // std::cout << "spring_constants" << std::endl;
    // std::cout << spring_constants << std::endl;

    // std::cout << "cart_spring_forces" << std::endl;
    // std::cout << cart_spring_forces << std::endl;

    // std::cout << "cart_damping_forces" << std::endl;
    // std::cout << cart_damping_forces << std::endl;

    // std::cout << "tau_repulsion" << std::endl;
    // std::cout << tau_repulsion << std::endl;

    
    std::cout << "dimension_2" << std::endl;
    for (const auto& element : dimension_2) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
    std::cout << "irregular_vector" << std::endl;
    for (const auto& element : irregular_vector) {
        std::cout << element << " ";
    }
    std::cout << std::endl;



    // std::cout << "Frep" << std::endl;
    // std::cout << Frep << std::endl;
    // std::cout << "Fpot_norms" << std::endl;
    // std::cout << Fpot_norms << std::endl;
    // std::cout << "Fpot_rot_norms" << std::endl;
    // std::cout << Fpot_rot_norms << std::endl;
    // std::cout << "norm_mask" << std::endl;
    // std::cout << norm_mask << std::endl;
    
    

    // double big_T = tau_impedance.array().abs().sum();
    // if (big_T > prev_big_T) {
    //   std::cout << "big_T" << std::endl;
    //   std::cout << big_T << std::endl;
    //   prev_big_T = big_T;
    // }
    // std::cout << "--------" << std::endl;
  }

  if (outcounter % 2/update_frequency == 0){
    if (csv_counter == 50*5){
      outFile1.open(path1);
      outFile2.open(path2);
      outFile3.open(path3);
    }

    if (outFile1.is_open()) {
        for (int i = 0; i < Frep.rows(); ++i) {
            for (int j = 0; j < Frep.cols(); ++j) {
                outFile1 << Frep(i, j);
                if (j < Frep.cols() - 1) {
                    outFile1 << ",";  // Add comma between values
                }
            }
            outFile1 << "\n";  // Newline at the end of each row
        }
    }
    if (outFile2.is_open()) {
        // outFile2 << Frep_norms.sum();
        // outFile2 << ",";
        for (int i = 0; i < tau_repulsion.rows(); ++i) {
            for (int j = 0; j < tau_repulsion.cols(); ++j) {
                outFile2 << tau_repulsion(i, j);
                if (j < tau_repulsion.cols() - 1) {
                    outFile2 << ",";  // Add comma between values
                }
            }
            outFile2 << "\n";  // Newline at the end of each row
        }
    }
    if (outFile3.is_open()) {
        // outFile3 << Fdamp_norms.sum();
        // outFile3 << ",";
        for (int i = 0; i < Fdamp_norms.rows(); ++i) {
            for (int j = 0; j < Fdamp_norms.cols(); ++j) {
                outFile3 << Fdamp_norms(i, j);
                if (j < Fdamp_norms.cols() - 1) {
                    outFile3 << ",";  // Add comma between values
                }
            }
            outFile3 << "\n";  // Newline at the end of each row
        }
    }

    csv_counter++;
    if (csv_counter == 500 * 40){
      outFile1.close();
      outFile2.close();
      outFile3.close();
    }
  }


  outcounter++;
  update_stiffness_and_references();
  return controller_interface::return_type::OK;
}
}

// namespace cartesian_impedance_control
#include "pluginlib/class_list_macros.hpp"
// NOLINTNEXTLINE
PLUGINLIB_EXPORT_CLASS(cartesian_impedance_control::CartesianImpedanceController,
                       controller_interface::ControllerInterface)
