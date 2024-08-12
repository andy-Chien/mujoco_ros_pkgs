/*********************************************************************
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2023, Bielefeld University
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Bielefeld University nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *********************************************************************/

/* Authors: David P. Leins */

#include <mujoco_ros/mujoco_env.h>

// #include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <geometry_msgs/msg/twist.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>

#include <mujoco_ros/util.h>

namespace mujoco_ros {
namespace mju = ::mujoco::sample_util;

bool MujocoEnv::verifyAdminHash(const std::string &hash)
{
	if (settings_.eval_mode) {
		RCLCPP_DEBUG(node_->get_logger(), "Evaluation mode is active. Checking hash validity");
		if (settings_.admin_hash != hash) {
			return false;
		}
		RCLCPP_DEBUG(node_->get_logger(), "Hash valid, request authorized.");
	}
	return true;
}

void MujocoEnv::setupServices()
{
	node_->create_service<>("add_two_ints", &add);
	service_servers_.emplace_back(node_->create_service<mujoco_ros_msgs::srv::SetPause>(
		"set_pause", &MujocoEnv::setPauseCB));
	service_servers_.emplace_back(node_->create_service<std_srvs::srv::Empty>(
		"shutdown", &MujocoEnv::shutdownCB));
	service_servers_.emplace_back(node_->create_service<mujoco_ros_msgs::srv::Reload>(
		"reload", &MujocoEnv::reloadCB));
	service_servers_.emplace_back(node_->create_service<std_srvs::srv::Empty>(
		"reset", &MujocoEnv::resetCB));
	service_servers_.emplace_back(node_->create_service<mujoco_ros_msgs::srv::SetBodyState>(
		"set_body_state", &MujocoEnv::setBodyStateCB));
	service_servers_.emplace_back(node_->create_service<mujoco_ros_msgs::srv::GetBodyState>(
		"get_body_state", &MujocoEnv::getBodyStateCB));
	service_servers_.emplace_back(node_->create_service<mujoco_ros_msgs::srv::SetGeomProperties>(
		"set_geom_properties", &MujocoEnv::setGeomPropertiesCB));
	service_servers_.emplace_back(node_->create_service<mujoco_ros_msgs::srv::GetGeomProperties>(
		"get_geom_properties", &MujocoEnv::getGeomPropertiesCB));
	service_servers_.emplace_back(node_->create_service<mujoco_ros_msgs::srv::SetEqualityConstraintParameters>(
		"set_eq_constraint_parameters", &MujocoEnv::setEqualityConstraintParametersArrayCB));
	service_servers_.emplace_back(node_->create_service<mujoco_ros_msgs::srv::GetEqualityConstraintParameters>(
		"get_eq_constraint_parameters", &MujocoEnv::getEqualityConstraintParametersArrayCB));
	service_servers_.emplace_back(node_->create_service<mujoco_ros_msgs::srv::GetStateUint>(
		"get_loading_request_state", &MujocoEnv::getStateUintCB));
	service_servers_.emplace_back(node_->create_service<mujoco_ros_msgs::srv::GetSimInfo>(
		"get_sim_info", &MujocoEnv::getSimInfoCB));
	service_servers_.emplace_back(node_->create_service<mujoco_ros_msgs::srv::SetFloat>(
		"set_rt_factor", &MujocoEnv::setRTFactorCB));
	service_servers_.emplace_back(node_->create_service<mujoco_ros_msgs::srv::GetPluginStats>(
		"get_plugin_stats", &MujocoEnv::getPluginStatsCB));
	service_servers_.emplace_back(node_->create_service<mujoco_ros_msgs::srv::SetGravity>(
		"set_gravity", &MujocoEnv::setGravityCB));
	service_servers_.emplace_back(node_->create_service<mujoco_ros_msgs::srv::GetGravity>(
		"get_gravity", &MujocoEnv::getGravityCB));
	service_servers_.emplace_back(node_->create_service<std_srvs::srv::Empty>(
	    "load_initial_joint_states", [&](const auto /*&req*/, auto /*&resp*/) {
			std::lock_guard<std::recursive_mutex> lock(physics_thread_mutex_);
			loadInitialJointStates();
			return true;
	}));

	using namespace std::placeholders;
	action_step_ = rclcpp_action::create_server<mujoco_ros_msgs_::action::Step>(
		node_,
		"step",
		std::bind(&MujocoEnv::action_step_handle_goal, this, _1, _2),
		std::bind(&MujocoEnv::action_step_handle_cancel, this, _1),
		std::bind(&MujocoEnv::action_step_handle_accepted, this, _1)
	);
}

using StepGoalHandle = rclcpp_action::ServerGoalHandle<mujoco_ros_msgs_::action::Step>;
rclcpp_action::GoalResponse MujocoEnv::action_step_handle_goal(
const rclcpp_action::GoalUUID & uuid,
std::shared_ptr<const mujoco_ros_msgs_::action::Step::Goal> goal)
{
	RCLCPP_INFO(node_->get_logger(), "Received goal request with order %d", goal->order);
	(void)uuid;
	if (settings_.env_steps_request.load() > 0 || settings_.run.load()) {
		RCLCPP_WARN(node_->get_logger(), 
			"Simulation is currently unpaused. Stepping makes no sense right now.");
		return rclcpp_action::GoalResponse::REJECT;
	}
	return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
}

rclcpp_action::CancelResponse MujocoEnv::action_step_handle_cancel(
const std::shared_ptr<StepGoalHandle> goal_handle)
{
	RCLCPP_INFO(node_->get_logger(), "Received request to cancel simulation step action goal");
	(void)goal_handle;
	settings_.env_steps_request.store(0);
	return rclcpp_action::CancelResponse::ACCEPT;
}

void MujocoEnv::action_step_handle_accepted(const std::shared_ptr<StepGoalHandle> goal_handle)
{
	using namespace std::placeholders;
	// this needs to return quickly to avoid blocking the executor, so spin up a new thread
	std::thread{std::bind(&MujocoEnv::action_step_execute, this, _1), goal_handle}.detach();
}

void MujocoEnv::action_step_execute(const std::shared_ptr<StepGoalHandle> goal_handle)
{
	RCLCPP_INFO(node_->get_logger(), "Executing simulation step action goal");
	const auto goal = goal_handle->get_goal();
	auto result = std::make_shared<mujoco_ros_msgs_::action::Step::Result>();
	auto feedback = std::make_shared<mujoco_ros_msgs_::action::Step::Feedback>();
	feedback->steps_left = goal->num_steps + util::as_unsigned(settings_.env_steps_request.load());
	settings_.env_steps_request.store(settings_.env_steps_request.load() + goal->num_steps);

	result->success = true;
	while (settings_.env_steps_request.load() > 0) {
		if (goal_handle->is_canceling() || !rclcpp::ok() || settings_.exit_request.load() > 0 ||
			settings_.load_request.load() > 0 || settings_.reset_request.load() > 0) {
			RCLCPP_WARN(node_->get_logger(), "Simulation step action goal canceled");
			result->success = false;
			settings_.env_steps_request.store(0);
			goal_handle->canceled(result);
			return;
		}

		feedback->steps_left = util::as_unsigned(settings_.env_steps_request.load());
		goal_handle->publish_feedback(feedback);
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}

	feedback->steps_left = util::as_unsigned(settings_.env_steps_request.load());
	goal_handle->publish_feedback(feedback);

	// Check if goal is done
	if (rclcpp::ok()) {
		result->sequence = sequence;
		goal_handle->succeed(result);
		RCLCPP_INFO(node_->get_logger(), "Goal succeeded");
	}
}

void MujocoEnv::runControlCbs()
{
	for (const auto &plugin : this->cb_ready_plugins_) {
		plugin->wrappedControlCallback(this->model_.get(), this->data_.get());
	}
}

void MujocoEnv::runPassiveCbs()
{
	for (const auto &plugin : this->cb_ready_plugins_) {
		plugin->wrappedPassiveCallback(this->model_.get(), this->data_.get());
	}
}

void MujocoEnv::runRenderCbs(mjvScene *scene)
{
	for (const auto &plugin : this->cb_ready_plugins_) {
		plugin->wrappedRenderCallback(this->model_.get(), this->data_.get(), scene);
	}
}

void MujocoEnv::runLastStageCbs()
{
	for (const auto &plugin : this->cb_ready_plugins_) {
		plugin->wrappedLastStageCallback(this->model_.get(), this->data_.get());
	}
}

bool MujocoEnv::setPauseCB(const std::shared_ptr<mujoco_ros_msgs::srv::SetPause::Request> req, std::shared_ptr<mujoco_ros_msgs::srv::SetPause::Response> resp)
{
	if (req->paused) {
		RCLCPP_DEBUG(node_->get_logger(), "Requested pause via ROS service");
	} else {
		RCLCPP_DEBUG(node_->get_logger(), "Requested unpause via ROS service");
	}
	resp->success = togglePaused(req->paused, req->admin_hash);
	return true;
}

bool MujocoEnv::shutdownCB(const std::shared_ptr<std_srvs::srv::Empty::Request> /*req*/, std::shared_ptr<std_srvs::srv::Empty::Response> /*resp*/)
{
	RCLCPP_DEBUG(node_->get_logger(), "Shutdown requested");
	settings_.exit_request.store(1);
	return true;
}

bool MujocoEnv::reloadCB(const std::shared_ptr<mujoco_ros_msgs::srv::Reload::Request> req, std::shared_ptr<mujoco_ros_msgs::srv::Reload::Response> resp)
{
	RCLCPP_DEBUG(node_->get_logger(), "Requested reload via ROS service");

	if (req->model.size() > kMaxFilenameLength) {
		RCLCPP_ERROR_STREAM(node_->get_logger(), "Model string too long. Max length: "
		                 << kMaxFilenameLength << " (got " << req->model.size()
		                 << "); Consider compiling with a larger value for kMaxFilenameLength");
		resp->success        = false;
		resp->status_message = "Model string too long (max: " + std::to_string(kMaxFilenameLength) + ")";
		return true;
	}
	mju::strcpy_arr(queued_filename_, req->model.c_str());

	settings_.load_request.store(2);

	while (getOperationalStatus() > 0) {
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}

	resp->success        = sim_state_.model_valid;
	resp->status_message = load_error_;

	return true;
}

bool MujocoEnv::resetCB(const std::shared_ptr<std_srvs::srv::Empty::Request> /*req*/, std::shared_ptr<std_srvs::srv::Empty::Response> /*resp*/)
{
	RCLCPP_DEBUG(node_->get_logger(), "Reset requested");
	settings_.reset_request.store(1);
	return true;
}

bool MujocoEnv::setBodyStateCB(const std::shared_ptr<mujoco_ros_msgs::srv::SetBodyState::Request> req,
                               std::shared_ptr<mujoco_ros_msgs::srv::SetBodyState::Response> resp)
{
	if (!verifyAdminHash(req->admin_hash)) {
		RCLCPP_ERROR(node_->get_logger(), "Hash mismatch, no permission to set body state!");
		resp->success = false;
		resp->status_message =
		    static_cast<decltype(resp->status_message)>("Hash mismatch, no permission to set body state!");
		return true;
	}

	std::string full_error_msg("");
	resp->success = true;

	int body_id = mj_name2id(model_.get(), mjOBJ_BODY, req->state.name.c_str());
	if (body_id == -1) {
		RCLCPP_WARN_STREAM(node_->get_logger(), "Could not find model (mujoco body) with name " << req->state.name << ". Trying to find geom...");
		int geom_id = mj_name2id(model_.get(), mjOBJ_GEOM, req->state.name.c_str());
		if (geom_id == -1) {
			std::string error_msg("Could not find model (not body nor geom) with name " + req->state.name);
			RCLCPP_WARN_STREAM(node_->get_logger(), error_msg);
			resp->status_message = error_msg;
			resp->success        = false;
			return true;
		}
		body_id = model_->geom_bodyid[geom_id];
		RCLCPP_WARN_STREAM(node_->get_logger(), "found body named '" << mj_id2name(model_.get(), mjOBJ_BODY, body_id) << "' as parent of geom '"
		                                     << req->state.name << "'");
	}

	if (req->set_mass) {
		std::lock_guard<std::recursive_mutex> lk_sim(physics_thread_mutex_);
		RCLCPP_DEBUG_STREAM(node_->get_logger(), "\tReplacing mass '" << model_->body_mass[body_id] << "' with new mass '" << req->state.mass
		                                      << "'");
		model_->body_mass[body_id] = req->state.mass;

		std::lock_guard<std::mutex> lk_render(offscreen_.render_mutex); // Prevent rendering the reset to q0
		mjtNum *qpos_tmp = mj_stackAllocNum(data_.get(), model_->nq);
		mju_copy(qpos_tmp, data_->qpos, model_->nq);
		RCLCPP_DEBUG(node_->get_logger(), "Copied current qpos state");
		mj_setConst(model_.get(), data_.get());
		RCLCPP_DEBUG(node_->get_logger(), "Reset constants because of mass change");
		mju_copy(data_->qpos, qpos_tmp, model_->nq);
		RCLCPP_DEBUG(node_->get_logger(), "Copied qpos state back to data");
	}

	int jnt_adr     = model_->body_jntadr[body_id];
	int jnt_type    = model_->jnt_type[jnt_adr];
	int num_jnt     = model_->body_jntnum[body_id];
	int jnt_qposadr = model_->jnt_qposadr[jnt_adr];
	int jnt_dofadr  = model_->jnt_dofadr[jnt_adr];

	geometry_msgs::msg::PoseStamped target_pose;
	geometry_msgs::msg::Twist target_twist;

	if (req->set_pose || req->set_twist || req->reset_qpos) {
		if (jnt_adr == -1) { // Check if body has joints
			std::string error_msg("Body has no joints, cannot move body!");
			RCLCPP_WARN_STREAM(node_->get_logger(), error_msg);
			full_error_msg += error_msg + '\n';
			resp->success = false;
		} else if (jnt_type != mjJNT_FREE) { // Only freejoints can be moved
			std::string error_msg("Body " + req->state.name +
			                      " has no joint of type 'freetype'. This service call does not support any other types!");
			RCLCPP_WARN_STREAM(node_->get_logger(), error_msg);
			full_error_msg += error_msg + '\n';
			resp->success = false;
		} else if (num_jnt > 1) {
			std::string error_msg("Body " + req->state.name + " has more than one joint ('" +
			                      std::to_string(model_->body_jntnum[body_id]) +
			                      "'), pose/twist changes to bodies with more than one joint are not supported!");
			RCLCPP_WARN_STREAM(node_->get_logger(), error_msg);
			full_error_msg += error_msg + '\n';
			resp->success = false;
		} else {
			// Lock mutex to prevent updating the body while a step is performed
			std::lock_guard<std::recursive_mutex> lk_sim(physics_thread_mutex_);
			geometry_msgs::msg::PoseStamped init_pose = req->state.pose;

			// Set freejoint position and quaternion
			if (req->set_pose && !req->reset_qpos) {
				bool valid_pose = true;
				if (!req->state.pose.header.frame_id.empty() && req->state.pose.header.frame_id != "world") {
					try {
						tf_bufferPtr_->transform<geometry_msgs::msg::PoseStamped>(req->state.pose, target_pose, "world");
					} catch (tf2::TransformException &ex) {
						RCLCPP_WARN_STREAM(node_->get_logger(), ex.what());
						full_error_msg +=
						    "Could not transform frame '" + req->state.pose.header.frame_id + "' to frame world" + '\n';
						resp->success = false;
						valid_pose   = false;
					}
				} else {
					target_pose = req->state.pose;
				}

				if (valid_pose) {
					mjtNum quat[4] = { target_pose.pose.orientation.w, target_pose.pose.orientation.x,
						                target_pose.pose.orientation.y, target_pose.pose.orientation.z };
					mju_normalize4(quat);

					RCLCPP_DEBUG_STREAM(node_->get_logger(), "Setting body pose to "
					                 << target_pose.pose.position.x << ", " << target_pose.pose.position.y << ", "
					                 << target_pose.pose.position.z << ", " << quat[0] << ", " << quat[1] << ", " << quat[2]
					                 << ", " << quat[3] << " (xyz wxyz)");

					data_->qpos[jnt_qposadr]     = target_pose.pose.position.x;
					data_->qpos[jnt_qposadr + 1] = target_pose.pose.position.y;
					data_->qpos[jnt_qposadr + 2] = target_pose.pose.position.z;
					data_->qpos[jnt_qposadr + 3] = quat[0];
					data_->qpos[jnt_qposadr + 4] = quat[1];
					data_->qpos[jnt_qposadr + 5] = quat[2];
					data_->qpos[jnt_qposadr + 6] = quat[3];
				}
			}

			auto req_set_twist = req->set_twist;
			auto req_state_twist = req->state_twist;

			if (req->reset_qpos && num_jnt > 0) {
				int num_dofs = 7; // Is always 7 because the joint is restricted to one joint of type freejoint
				RCLCPP_WARN_EXPRESSION(node_->get_logger(), req->set_pose,
				              "set_pose and reset_qpos were both passed. reset_qpos will overwrite the custom pose!");
				RCLCPP_DEBUG(node_->get_logger(), "Resetting body qpos");
				mju_copy(data_->qpos + model_->jnt_qposadr[jnt_adr], model_->qpos0 + model_->jnt_qposadr[jnt_adr],
				         num_dofs);
				if (!req->set_twist) {
					// Reset twist if no desired twist is given (default twist is 0 0 0 0 0 0)
					req_set_twist   = true;
					req_state_twist = geometry_msgs::msg::TwistStamped();
				}
			}
			// Set freejoint twist
			if (req_set_twist) {
				// Only pose can be transformed. Twist will be ignored!
				if (!req_state_twist.header.frame_id.empty() && req_state_twist.header.frame_id != "world") {
					std::string error_msg("Transforming twists from other frames is not supported! Not setting twist.");
					RCLCPP_WARN_STREAM(node_->get_logger(), error_msg);
					full_error_msg += error_msg + '\n';
					resp->success = false;
				} else {
					RCLCPP_DEBUG_STREAM(node_->get_logger(), "Setting body twist to "
					                 << req_state_twist.twist.linear.x << ", " << req_state_twist.twist.linear.y << ", "
					                 << req_state_twist.twist.linear.z << ", " << req_state_twist.twist.angular.x << ", "
					                 << req_state_twist.twist.angular.y << ", " << req_state_twist.twist.angular.z
					                 << " (xyz rpy)");
					data_->qvel[jnt_dofadr]     = req_state_twist.twist.linear.x;
					data_->qvel[jnt_dofadr + 1] = req_state_twist.twist.linear.y;
					data_->qvel[jnt_dofadr + 2] = req_state_twist.twist.linear.z;
					data_->qvel[jnt_dofadr + 3] = req_state_twist.twist.angular.x;
					data_->qvel[jnt_dofadr + 4] = req_state_twist.twist.angular.y;
					data_->qvel[jnt_dofadr + 5] = req_state_twist.twist.angular.z;
				}
			}
		}
	}

	resp->status_message = full_error_msg;
	return true;
}

bool MujocoEnv::getBodyStateCB(const std::shared_ptr<mujoco_ros_msgs::srv::GetBodyState::Request> req,
                               std::shared_ptr<mujoco_ros_msgs::srv::GetBodyState::Response> resp)
{
	if (!verifyAdminHash(req->admin_hash)) {
		RCLCPP_ERROR(node_->get_logger(), "Hash mismatch, no permission to get body state!");
		resp->status_message =
		    static_cast<decltype(resp->status_message)>("Hash mismatch, no permission to get body state!");
		resp->success = false;
		return true;
	}

	resp->success = true;

	int body_id = mj_name2id(model_.get(), mjOBJ_BODY, req->name.c_str());
	if (body_id == -1) {
		RCLCPP_WARN_STREAM(node_->get_logger(), "Could not find model (mujoco body) with name " << req->name << ". Trying to find geom...");
		int geom_id = mj_name2id(model_.get(), mjOBJ_GEOM, req->name.c_str());
		if (geom_id == -1) {
			std::string error_msg("Could not find model (not body nor geom) with name " + req->name);
			RCLCPP_WARN_STREAM(node_->get_logger(), error_msg);
			resp->status_message = error_msg;
			resp->success        = false;
			return true;
		}
		body_id = model_->geom_bodyid[geom_id];
		RCLCPP_WARN_STREAM(node_->get_logger(), "found body named '" << mj_id2name(model_.get(), mjOBJ_BODY, body_id) << "' as parent of geom '"
		                                     << req->name << "'");
	}

	resp->state.name = mj_id2name(model_.get(), mjOBJ_BODY, body_id);
	resp->state.mass = static_cast<decltype(resp->state.mass)>(model_->body_mass[body_id]);

	int jnt_adr     = model_->body_jntadr[body_id];
	int jnt_type    = model_->jnt_type[jnt_adr];
	int num_jnt     = model_->body_jntnum[body_id];
	int jnt_qposadr = model_->jnt_qposadr[jnt_adr];
	int jnt_dofadr  = model_->jnt_dofadr[jnt_adr];

	geometry_msgs::msg::PoseStamped target_pose;
	geometry_msgs::msg::Twist target_twist;

	// Stop sim to get data out of the same point in time
	std::lock_guard<std::recursive_mutex> lk_sim(physics_thread_mutex_);
	if (jnt_adr == -1 || jnt_type != mjJNT_FREE || num_jnt > 1) {
		resp->state.pose.header             = std_msgs::Header();
		resp->state.pose.header.frame_id    = "world";
		resp->state.pose.pose.position.x    = data_->xpos[body_id * 3];
		resp->state.pose.pose.position.y    = data_->xpos[body_id * 3 + 1];
		resp->state.pose.pose.position.z    = data_->xpos[body_id * 3 + 2];
		resp->state.pose.pose.orientation.w = data_->xquat[body_id * 3];
		resp->state.pose.pose.orientation.x = data_->xquat[body_id * 3 + 1];
		resp->state.pose.pose.orientation.y = data_->xquat[body_id * 3 + 2];
		resp->state.pose.pose.orientation.z = data_->xquat[body_id * 3 + 3];

		resp->state.twist.header          = std_msgs::Header();
		resp->state.twist.header.frame_id = "world";
		resp->state.twist.twist.linear.x  = data_->cvel[body_id * 6];
		resp->state.twist.twist.linear.y  = data_->cvel[body_id * 6 + 1];
		resp->state.twist.twist.linear.z  = data_->cvel[body_id * 6 + 2];
		resp->state.twist.twist.angular.x = data_->cvel[body_id * 6 + 3];
		resp->state.twist.twist.angular.y = data_->cvel[body_id * 6 + 4];
		resp->state.twist.twist.angular.z = data_->cvel[body_id * 6 + 5];
	} else {
		resp->state.pose.header             = std_msgs::Header();
		resp->state.pose.header.frame_id    = "world";
		resp->state.pose.pose.position.x    = data_->qpos[jnt_qposadr];
		resp->state.pose.pose.position.y    = data_->qpos[jnt_qposadr + 1];
		resp->state.pose.pose.position.z    = data_->qpos[jnt_qposadr + 2];
		resp->state.pose.pose.orientation.w = data_->qpos[jnt_qposadr + 3];
		resp->state.pose.pose.orientation.x = data_->qpos[jnt_qposadr + 4];
		resp->state.pose.pose.orientation.y = data_->qpos[jnt_qposadr + 5];
		resp->state.pose.pose.orientation.z = data_->qpos[jnt_qposadr + 6];

		resp->state.twist.header          = std_msgs::Header();
		resp->state.twist.header.frame_id = "world";
		resp->state.twist.twist.linear.x  = data_->qvel[jnt_dofadr];
		resp->state.twist.twist.linear.y  = data_->qvel[jnt_dofadr + 1];
		resp->state.twist.twist.linear.z  = data_->qvel[jnt_dofadr + 2];
		resp->state.twist.twist.angular.x = data_->qvel[jnt_dofadr + 3];
		resp->state.twist.twist.angular.y = data_->qvel[jnt_dofadr + 4];
		resp->state.twist.twist.angular.z = data_->qvel[jnt_dofadr + 5];
	}

	return true;
}

bool MujocoEnv::setGravityCB(const std::shared_ptr<mujoco_ros_msgs::srv::SetGravity::Request> req, std::shared_ptr<mujoco_ros_msgs::srv::SetGravity::Response> resp)
{
	if (!verifyAdminHash(req->admin_hash)) {
		RCLCPP_ERROR(node_->get_logger(), "Hash mismatch, no permission to set gravity!");
		resp->status_message = static_cast<decltype(resp->status_message)>("Hash mismatch, no permission to set gravity!");
		resp->success        = false;
		return true;
	}

	// Lock mutex to set data within one step
	std::lock_guard<std::recursive_mutex> lk_sim(physics_thread_mutex_);
	for (size_t i = 0; i < 3; ++i) {
		model_->opt.gravity[i] = req->gravity[i];
	}
	resp->success = true;
	return true;
}

bool MujocoEnv::getGravityCB(const std::shared_ptr<mujoco_ros_msgs::srv::GetGravity::Request> req, std::shared_ptr<mujoco_ros_msgs::srv::GetGravity::Response> resp)
{
	if (!verifyAdminHash(req->admin_hash)) {
		RCLCPP_ERROR(node_->get_logger(), "Hash mismatch, no permission to get gravity!");
		resp->status_message = static_cast<decltype(resp->status_message)>("Hash mismatch, no permission to get gravity!");
		resp->success        = false;
		return true;
	}

	// Lock mutex to get data within one step
	std::lock_guard<std::recursive_mutex> lk_sim(physics_thread_mutex_);
	for (size_t i = 0; i < 3; ++i) {
		resp->gravity[i] = model_->opt.gravity[i];
	}
	resp->success = true;
	return true;
}

bool MujocoEnv::setGeomPropertiesCB(const std::shared_ptr<mujoco_ros_msgs::srv::SetGeomProperties::Request> req,
                                    std::shared_ptr<mujoco_ros_msgs::srv::SetGeomProperties::Response> resp)
{
	if (!verifyAdminHash(req->admin_hash)) {
		RCLCPP_ERROR(node_->get_logger(), "Hash mismatch, no permission to set geom properties!");
		resp->status_message =
		    static_cast<decltype(resp->status_message)>("Hash mismatch, no permission to set geom properties!");
		resp->success = false;
		return true;
	}

	int geom_id = mj_name2id(model_.get(), mjOBJ_GEOM, req->properties.name.c_str());
	if (geom_id == -1) {
		std::string error_msg("Could not find model (mujoco geom) with name " + req->properties.name);
		RCLCPP_WARN_STREAM(node_->get_logger(), error_msg);
		resp->status_message = error_msg;
		resp->success        = false;
		return true;
	}

	int body_id = model_->geom_bodyid[geom_id];

	// Lock mutex to prevent updating the body while a step is performed
	std::lock_guard<std::recursive_mutex> lk_sim(physics_thread_mutex_);

	RCLCPP_DEBUG_STREAM(node_->get_logger(), "Changing properties of geom '" << req->properties.name.c_str() << "' ...");
	if (req->set_mass) {
		RCLCPP_DEBUG_STREAM(node_->get_logger(), "\tReplacing mass '" << model_->body_mass[body_id] << "' with new mass '"
		                                      << req->properties.body_mass << "'");
		model_->body_mass[body_id] = req->properties.body_mass;
	}
	if (req->set_friction) {
		RCLCPP_DEBUG_STREAM(node_->get_logger(), "\tReplacing friction '"
		                 << model_->geom_friction[geom_id * 3] << ", " << model_->geom_friction[geom_id * 3 + 1] << ", "
		                 << model_->geom_friction[geom_id * 3 + 2] << "' with new mass '" << req->properties.friction_slide
		                 << ", " << req->properties.friction_spin << ", " << req->properties.friction_roll << "'");
		model_->geom_friction[geom_id * 3]     = req->properties.friction_slide;
		model_->geom_friction[geom_id * 3 + 1] = req->properties.friction_spin;
		model_->geom_friction[geom_id * 3 + 2] = req->properties.friction_roll;
	}
	if (req->set_type) {
		RCLCPP_DEBUG_STREAM(node_->get_logger(), "\tReplacing type '" << model_->geom_type[geom_id] << "' with new type '" << req->properties.type
		                                      << "'");
		model_->geom_type[geom_id] = req->properties.type.value;
	}

	if (req->set_size) {
		if (static_cast<mjtNum>(req->properties.size_0 * req->properties.size_1 * req->properties.size_2) >
		    model_->geom_size[geom_id * 3] * model_->geom_size[geom_id * 3 + 1] * model_->geom_size[geom_id * 3 + 2]) {
			RCLCPP_WARN_STREAM(node_->get_logger(), "New geom size is bigger than the old size. AABBs are not recomputed, this might cause "
			                "incorrect collisions!");
		}

		RCLCPP_DEBUG_STREAM(node_->get_logger(), "\tReplacing size '"
		                 << model_->geom_size[geom_id * 3] << ", " << model_->geom_size[geom_id * 3 + 1] << ", "
		                 << model_->geom_size[geom_id * 3 + 2] << "' with new size '" << req->properties.size_0 << ", "
		                 << req->properties.size_1 << ", " << req->properties.size_2 << "'");
		model_->geom_size[geom_id * 3]     = req->properties.size_0;
		model_->geom_size[geom_id * 3 + 1] = req->properties.size_1;
		model_->geom_size[geom_id * 3 + 2] = req->properties.size_2;

		mj_forward(model_.get(), data_.get());
	}

	if (req->set_type || req->set_mass) {
		std::lock_guard<std::mutex> lk_render(offscreen_.render_mutex); // Prevent rendering the reset to q0

		mjtNum *qpos_tmp = mj_stackAllocNum(data_.get(), model_->nq);
		mju_copy(qpos_tmp, data_->qpos, model_->nq);
		RCLCPP_DEBUG(node_->get_logger(), "Copied current qpos state");
		mj_setConst(model_.get(), data_.get());
		RCLCPP_DEBUG(node_->get_logger(), "Reset constants");
		mju_copy(data_->qpos, qpos_tmp, model_->nq);
		RCLCPP_DEBUG(node_->get_logger(), "Copied qpos state back to data");
	}

	notifyGeomChanged(geom_id);

	resp->success = true;
	return true;
}

bool MujocoEnv::getGeomPropertiesCB(const std::shared_ptr<mujoco_ros_msgs::srv::GetGeomProperties::Request> req,
                                    std::shared_ptr<mujoco_ros_msgs::srv::GetGeomProperties::Response> resp)
{
	if (!verifyAdminHash(req->admin_hash)) {
		RCLCPP_ERROR(node_->get_logger(), "Hash mismatch, no permission to get geom properties!");
		resp->status_message =
		    static_cast<decltype(resp->status_message)>("Hash mismatch, no permission to get geom properties!");
		resp->success = false;
		return true;
	}

	int geom_id = mj_name2id(model_.get(), mjOBJ_GEOM, req->geom_name.c_str());
	if (geom_id == -1) {
		std::string error_msg("Could not find model (mujoco geom) with name " + req->geom_name);
		RCLCPP_WARN_STREAM(node_->get_logger(), error_msg);
		resp->status_message = error_msg;
		resp->success        = false;
		return true;
	}

	int body_id = model_->geom_bodyid[geom_id];

	// Lock mutex to get data within one step
	std::lock_guard<std::recursive_mutex> lk_sim(physics_thread_mutex_);
	resp->properties.name      = req->geom_name;
	resp->properties.body_mass = static_cast<decltype(resp->properties.body_mass)>(model_->body_mass[body_id]);
	resp->properties.friction_slide =
	    static_cast<decltype(resp->properties.friction_slide)>(model_->geom_friction[geom_id * 3]);
	resp->properties.friction_spin =
	    static_cast<decltype(resp->properties.friction_spin)>(model_->geom_friction[geom_id * 3 + 1]);
	resp->properties.friction_roll =
	    static_cast<decltype(resp->properties.friction_roll)>(model_->geom_friction[geom_id * 3 + 2]);

	resp->properties.type.value = static_cast<decltype(resp->properties.type.value)>(model_->geom_type[geom_id]);

	resp->properties.size_0 = static_cast<decltype(resp->properties.size_0)>(model_->geom_size[geom_id * 3]);
	resp->properties.size_1 = static_cast<decltype(resp->properties.size_1)>(model_->geom_size[geom_id * 3 + 1]);
	resp->properties.size_2 = static_cast<decltype(resp->properties.size_2)>(model_->geom_size[geom_id * 3 + 2]);

	resp->success = true;
	return true;
}

bool MujocoEnv::setEqualityConstraintParameters(const mujoco_ros_msgs::msg::EqualityConstraintParameters &parameters)
{
	// look up equality constraint by name
	RCLCPP_DEBUG_STREAM(node_->get_logger(), "Looking up eqc by name '" << parameters.name << "'");
	int eq_id = mj_name2id(model_.get(), mjOBJ_EQUALITY, parameters.name.c_str());
	if (eq_id != -1) {
		RCLCPP_DEBUG_STREAM(node_->get_logger(), "Found eqc by name '" << parameters.name << "'");
		int id1, id2;
		switch (parameters.type.value) {
			case mjEQ_TENDON:
				id1 = mj_name2id(model_.get(), mjOBJ_TENDON, parameters.element1.c_str());
				if (id1 != -1) {
					model_->eq_obj1id[eq_id] = id1;
				}
				if (!parameters.element2.empty()) {
					id2 = mj_name2id(model_.get(), mjOBJ_TENDON, parameters.element2.c_str());
					if (id2 != -1) {
						model_->eq_obj2id[eq_id] = id2;
					}
				}
				model_->eq_data[eq_id * mjNEQDATA]     = parameters.polycoef[0];
				model_->eq_data[eq_id * mjNEQDATA + 1] = parameters.polycoef[1];
				model_->eq_data[eq_id * mjNEQDATA + 2] = parameters.polycoef[2];
				model_->eq_data[eq_id * mjNEQDATA + 3] = parameters.polycoef[3];
				model_->eq_data[eq_id * mjNEQDATA + 4] = parameters.polycoef[4];
				break;
			case mjEQ_WELD:
				id1 = mj_name2id(model_.get(), mjOBJ_XBODY, parameters.element1.c_str());
				if (id1 != -1) {
					model_->eq_obj1id[eq_id] = id1;
				}
				if (!parameters.element2.empty()) {
					id2 = mj_name2id(model_.get(), mjOBJ_XBODY, parameters.element2.c_str());
					if (id2 != -1) {
						model_->eq_obj2id[eq_id] = id2;
					}
				}
				model_->eq_data[eq_id * mjNEQDATA]      = parameters.anchor.x;
				model_->eq_data[eq_id * mjNEQDATA + 1]  = parameters.anchor.y;
				model_->eq_data[eq_id * mjNEQDATA + 2]  = parameters.anchor.z;
				model_->eq_data[eq_id * mjNEQDATA + 3]  = parameters.relpose.position.x;
				model_->eq_data[eq_id * mjNEQDATA + 4]  = parameters.relpose.position.y;
				model_->eq_data[eq_id * mjNEQDATA + 5]  = parameters.relpose.position.z;
				model_->eq_data[eq_id * mjNEQDATA + 6]  = parameters.relpose.orientation.w;
				model_->eq_data[eq_id * mjNEQDATA + 7]  = parameters.relpose.orientation.x;
				model_->eq_data[eq_id * mjNEQDATA + 8]  = parameters.relpose.orientation.y;
				model_->eq_data[eq_id * mjNEQDATA + 9]  = parameters.relpose.orientation.z;
				model_->eq_data[eq_id * mjNEQDATA + 10] = parameters.torquescale;
				break;
			case mjEQ_JOINT:
				id1 = mj_name2id(model_.get(), mjOBJ_JOINT, parameters.element1.c_str());
				if (id1 != -1) {
					model_->eq_obj1id[eq_id] = id1;
				}
				if (!parameters.element2.empty()) {
					id2 = mj_name2id(model_.get(), mjOBJ_JOINT, parameters.element2.c_str());
					if (id2 != -1) {
						model_->eq_obj2id[eq_id] = id2;
					}
				}
				model_->eq_data[eq_id * mjNEQDATA]     = parameters.polycoef[0];
				model_->eq_data[eq_id * mjNEQDATA + 1] = parameters.polycoef[1];
				model_->eq_data[eq_id * mjNEQDATA + 2] = parameters.polycoef[2];
				model_->eq_data[eq_id * mjNEQDATA + 3] = parameters.polycoef[3];
				model_->eq_data[eq_id * mjNEQDATA + 4] = parameters.polycoef[4];
				break;
			case mjEQ_CONNECT:
				id1 = mj_name2id(model_.get(), mjOBJ_XBODY, parameters.element1.c_str());
				if (id1 != -1) {
					model_->eq_obj1id[eq_id] = id1;
				}
				if (!parameters.element2.empty()) {
					id2 = mj_name2id(model_.get(), mjOBJ_XBODY, parameters.element2.c_str());

					if (id2 != -1) {
						model_->eq_obj2id[eq_id] = id2;
					}
				}
				model_->eq_data[eq_id * mjNEQDATA]     = parameters.anchor.x;
				model_->eq_data[eq_id * mjNEQDATA + 1] = parameters.anchor.y;
				model_->eq_data[eq_id * mjNEQDATA + 2] = parameters.anchor.z;
				break;
			default:
				break;
		}
		data_->eq_active[eq_id]               = parameters.active;
		model_->eq_solimp[eq_id * mjNIMP]     = parameters.solverParameters.dmin;
		model_->eq_solimp[eq_id * mjNIMP + 1] = parameters.solverParameters.dmax;
		model_->eq_solimp[eq_id * mjNIMP + 2] = parameters.solverParameters.width;
		model_->eq_solimp[eq_id * mjNIMP + 3] = parameters.solverParameters.midpoint;
		model_->eq_solimp[eq_id * mjNIMP + 4] = parameters.solverParameters.power;
		model_->eq_solref[eq_id * mjNREF]     = parameters.solverParameters.timeconst;
		model_->eq_solref[eq_id * mjNREF + 1] = parameters.solverParameters.dampratio;
		return true;
	}
	RCLCPP_WARN_STREAM(node_->get_logger(), "Could not find specified equality constraint with name '" << parameters.name << "'");
	return false;
}

bool MujocoEnv::setEqualityConstraintParametersArrayCB(const std::shared_ptr<mujoco_ros_msgs::srv::SetEqualityConstraintParameters::Request> req,
                                                       std::shared_ptr<mujoco_ros_msgs::srv::SetEqualityConstraintParameters::Response> resp)
{
	if (!verifyAdminHash(req->admin_hash)) {
		RCLCPP_ERROR(node_->get_logger(), "Hash mismatch, no permission to set equality constraints!");
		resp->status_message =
		    static_cast<decltype(resp->status_message)>("Hash mismatch, no permission to set equality constraints!");
		resp->success = false;
		return true;
	}
	resp->success = true;

	bool failed_any    = false;
	bool succeeded_any = false;
	for (const auto &parameters : req->parameters) {
		bool success  = setEqualityConstraintParameters(parameters);
		failed_any    = (failed_any || !success);
		succeeded_any = (succeeded_any || success);
	}

	if (succeeded_any && failed_any) {
		resp->status_message = static_cast<decltype(resp->status_message)>("Not all constraints could be set");
		resp->success        = false;
	} else if (failed_any) {
		resp->status_message = static_cast<decltype(resp->status_message)>("Could not set any constraints");
		resp->success        = false;
	}

	return true;
}

bool MujocoEnv::getEqualityConstraintParameters(mujoco_ros_msgs::msg::EqualityConstraintParameters &parameters)
{
	RCLCPP_DEBUG_STREAM(node_->get_logger(), "Looking up Eq Constraint '" << parameters.name << "'");
	// look up equality constraint by name
	int eq_id = mj_name2id(model_.get(), mjOBJ_EQUALITY, parameters.name.c_str());
	if (eq_id != -1) {
		RCLCPP_DEBUG(node_->get_logger(), "Found Eq Constraint");
		parameters.type.value = model_->eq_type[eq_id];

		std::vector<float> polycoef = std::vector<float>(5);

		switch (model_->eq_type[eq_id]) {
			case mjEQ_CONNECT:
				parameters.element1 = mj_id2name(model_.get(), mjOBJ_JOINT, model_->eq_obj1id[eq_id]);
				if (mj_id2name(model_.get(), mjOBJ_JOINT, model_->eq_obj2id[eq_id])) {
					parameters.element2 = mj_id2name(model_.get(), mjOBJ_BODY, model_->eq_obj2id[eq_id]);
				}
				break;
			case mjEQ_WELD:
				parameters.element1 = mj_id2name(model_.get(), mjOBJ_BODY, model_->eq_obj1id[eq_id]);
				if (mj_id2name(model_.get(), mjOBJ_BODY, model_->eq_obj2id[eq_id])) {
					parameters.element2 = mj_id2name(model_.get(), mjOBJ_BODY, model_->eq_obj2id[eq_id]);
				}
				parameters.anchor.x              = model_->eq_data[eq_id * mjNEQDATA];
				parameters.anchor.y              = model_->eq_data[eq_id * mjNEQDATA + 1];
				parameters.anchor.z              = model_->eq_data[eq_id * mjNEQDATA + 2];
				parameters.relpose.position.x    = model_->eq_data[eq_id * mjNEQDATA + 3];
				parameters.relpose.position.y    = model_->eq_data[eq_id * mjNEQDATA + 4];
				parameters.relpose.position.z    = model_->eq_data[eq_id * mjNEQDATA + 5];
				parameters.relpose.orientation.w = model_->eq_data[eq_id * mjNEQDATA + 6];
				parameters.relpose.orientation.x = model_->eq_data[eq_id * mjNEQDATA + 7];
				parameters.relpose.orientation.y = model_->eq_data[eq_id * mjNEQDATA + 8];
				parameters.relpose.orientation.z = model_->eq_data[eq_id * mjNEQDATA + 9];
				parameters.torquescale           = model_->eq_data[eq_id * mjNEQDATA + 10];
				break;
			case mjEQ_JOINT:
				parameters.element1 = mj_id2name(model_.get(), mjOBJ_JOINT, model_->eq_obj1id[eq_id]);
				if (mj_id2name(model_.get(), mjOBJ_JOINT, model_->eq_obj2id[eq_id])) {
					parameters.element2 = mj_id2name(model_.get(), mjOBJ_JOINT, model_->eq_obj2id[eq_id]);
				}
				parameters.polycoef = { model_->eq_data[eq_id * mjNEQDATA], model_->eq_data[eq_id * mjNEQDATA + 1],
					                     model_->eq_data[eq_id * mjNEQDATA + 2], model_->eq_data[eq_id * mjNEQDATA + 3],
					                     model_->eq_data[eq_id * mjNEQDATA + 4] };
				break;
			case mjEQ_TENDON:
				parameters.element1 = mj_id2name(model_.get(), mjOBJ_TENDON, model_->eq_obj1id[eq_id]);
				if (mj_id2name(model_.get(), mjOBJ_TENDON, model_->eq_obj2id[eq_id])) {
					parameters.element2 = mj_id2name(model_.get(), mjOBJ_TENDON, model_->eq_obj2id[eq_id]);
				}
				parameters.polycoef = { model_->eq_data[eq_id * mjNEQDATA], model_->eq_data[eq_id * mjNEQDATA + 1],
					                     model_->eq_data[eq_id * mjNEQDATA + 2], model_->eq_data[eq_id * mjNEQDATA + 3],
					                     model_->eq_data[eq_id * mjNEQDATA + 4] };
				break;
			default:
				break;
		}
		parameters.active                     = data_->eq_active[eq_id];
		parameters.solverParameters.dmin      = model_->eq_solimp[eq_id * mjNIMP];
		parameters.solverParameters.dmax      = model_->eq_solimp[eq_id * mjNIMP + 1];
		parameters.solverParameters.width     = model_->eq_solimp[eq_id * mjNIMP + 2];
		parameters.solverParameters.midpoint  = model_->eq_solimp[eq_id * mjNIMP + 3];
		parameters.solverParameters.power     = model_->eq_solimp[eq_id * mjNIMP + 4];
		parameters.solverParameters.timeconst = model_->eq_solref[eq_id * mjNREF];
		parameters.solverParameters.dampratio = model_->eq_solref[eq_id * mjNREF + 1];
		return true;
	}
	RCLCPP_WARN_STREAM(node_->get_logger(), "Could not find equality constraint named '" << parameters.name << "'");
	return false;
}

bool MujocoEnv::getEqualityConstraintParametersArrayCB(const std::shared_ptr<mujoco_ros_msgs::srv::GetEqualityConstraintParameters::Request> req,
                                                       std::shared_ptr<mujoco_ros_msgs::srv::GetEqualityConstraintParameters::Response> resp)
{
	if (!verifyAdminHash(req->admin_hash)) {
		RCLCPP_ERROR(node_->get_logger(), "Hash mismatch, no permission to get equality constraints!");
		resp->status_message =
		    static_cast<decltype(resp->status_message)>("Hash mismatch, no permission to get equality constraints!");
		resp->success = false;
		return true;
	}
	resp->success = true;

	bool failed_any    = false;
	bool succeeded_any = false;
	for (const auto &name : req->names) {
		std::shared_ptr<mujoco_ros_msgs::srv::EqualityConstraintParameters eqc;> 	eqc.name     = name;
		bool success = getEqualityConstraintParameters(eqc);

		failed_any    = (failed_any || !success);
		succeeded_any = (succeeded_any || success);
		if (success) {
			resp->parameters.emplace_back(eqc);
		}
	}

	if (succeeded_any && failed_any) {
		resp->status_message = static_cast<decltype(resp->status_message)>("Not all constraints could be fetched");
		resp->success        = false;
	} else if (failed_any) {
		resp->status_message = static_cast<decltype(resp->status_message)>("Could not fetch any constraints");
		resp->success        = false;
	}

	return true;
}

bool MujocoEnv::getStateUintCB(const std::shared_ptr<mujoco_ros_msgs::srv::GetStateUint::Request>  /*req*/,
                               std::shared_ptr<mujoco_ros_msgs::srv::GetStateUint::Response> resp)
{
	int status       = getOperationalStatus();
	resp->state.value = static_cast<decltype(resp->state.value)>(status);

	std::string description;
	if (status == 0)
		description = "Sim ready";
	else if (status == 1)
		description = "Loading in progress";
	else if (status >= 2)
		description = "Loading issued";
	resp->state.description = description;
	return true;
}

bool MujocoEnv::getSimInfoCB(const std::shared_ptr<mujoco_ros_msgs::srv::GetSimInfo::Request>  /*req*/,
                             std::shared_ptr<mujoco_ros_msgs::srv::GetSimInfo::Response> resp)
{
	std::shared_ptr<mujoco_ros_msgs::srv::GetStateUint state_srv;> getStateUintCB(state_srv.request, state_srv.response);

	resp->state.model_path        = filename_;
	resp->state.model_valid       = sim_state_.model_valid;
	resp->state.load_count        = sim_state_.load_count;
	resp->state.loading_state     = state_srv.response.state;
	resp->state.paused            = !settings_.run.load();
	resp->state.pending_sim_steps = settings_.env_steps_request.load();
	resp->state.rt_measured       = 1.f / sim_state_.measured_slowdown;
	resp->state.rt_setting        = percentRealTime[settings_.real_time_index] / 100.f;
	return true;
}

// Helper function to retrieve the real-time factor closest to the requested value
// adapted from https://www.geeksforgeeks.org/find-closest-number-array/
float findClosestRecursive(const float arr[], uint left, uint right, float target)
{
	if (left == right) {
		return arr[left];
	}

	uint mid            = (left + right) / 2;
	float left_closest  = findClosestRecursive(arr, left, mid, target);
	float right_closest = findClosestRecursive(arr, mid + 1, right, target);

	if (abs(left_closest - target) <= abs(right_closest - target)) {
		return left_closest;
	} else {
		return right_closest;
	}
}

bool MujocoEnv::setRTFactorCB(const std::shared_ptr<mujoco_ros_msgs::srv::SetFloat::Request> req, std::shared_ptr<mujoco_ros_msgs::srv::SetFloat::Response> resp)
{
	if (!verifyAdminHash(req->admin_hash)) {
		RCLCPP_ERROR(node_->get_logger(), "Hash mismatch, no permission to set real-time factor!");
		resp->success = false;
		return true;
	}
	resp->success = true;

	if (req->value < 0) {
		settings_.real_time_index = 0;
		settings_.speed_changed   = true;
		resp->success              = true;
		return true;
	}

	// find value closest to requested
	size_t num_clicks = sizeof(percentRealTime) / sizeof(percentRealTime[0]);
	float closest =
	    findClosestRecursive(percentRealTime, 1, num_clicks - 1,
	                         100.f * static_cast<float>(req->value)); // start at 1 to not go to unbound mode if the value
	                                                                 // is too small (already handled above)

	RCLCPP_WARN_STREAM_EXPRESSION(node_->get_logger(), fabs(closest / 100.f - static_cast<float>(req->value)) > 0.001f,
	                     "Requested factor '" << req->value
	                                          << "' not available, setting to closest available: " << closest / 100.f);

	// get index of closest value
	auto it                   = std::find(std::next(std::begin(percentRealTime)), std::end(percentRealTime), closest);
	settings_.real_time_index = std::distance(std::begin(percentRealTime), it);
	settings_.speed_changed   = true;
	return true;
}

bool MujocoEnv::getPluginStatsCB(const std::shared_ptr<mujoco_ros_msgs::srv::GetPluginStats::Request>  /*req*/,
                                 std::shared_ptr<mujoco_ros_msgs::srv::GetPluginStats::Response> resp)
{
	// Lock mutex to get data within one step
	std::lock_guard<std::recursive_mutex> lk_sim(physics_thread_mutex_);
	for (const auto &plugin : plugins_) {
		std::shared_ptr<mujoco_ros_msgs::srv::PluginStats stats;> 	stats.plugin_type             = plugin->type_;
		stats.load_time               = plugin->load_time_;
		stats.reset_time              = plugin->reset_time_;
		stats.ema_steptime_control    = plugin->ema_steptime_control_;
		stats.ema_steptime_passive    = plugin->ema_steptime_passive_;
		stats.ema_steptime_render     = plugin->ema_steptime_render_;
		stats.ema_steptime_last_stage = plugin->ema_steptime_last_stage_;
		resp->stats.emplace_back(stats);
	}
	return true;
}

} // namespace mujoco_ros
