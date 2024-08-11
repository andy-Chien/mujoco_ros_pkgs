// Copyright 2021 DeepMind Technologies Limited
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

#pragma once

#include <thread>
#include <boost/thread.hpp>

#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/empty.hpp>
#include <rosgraph_msgs/msg/clock.hpp>
#include <tf2_ros/msg/transform_listener.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/msg/static_transform_broadcaster.hpp>

#include "mujoco_ros/viewer.h"
#include "mujoco_ros/common_types.h"
#include "mujoco_ros/plugin_utils.h"
#include "mujoco_ros/glfw_adapter.h"
#include "mujoco_ros/glfw_dispatch.h"

#include "mujoco_ros_msgs/msg/plugin_stats.hpp"
#include "mujoco_ros_msgs/msg/equality_constraint_parameters.hpp"

#include "mujoco_ros_msgs/srv/reload.hpp"
#include "mujoco_ros_msgs/srv/set_pause.hpp"
#include "mujoco_ros_msgs/srv/set_float.hpp"
#include "mujoco_ros_msgs/srv/set_gravity.hpp"
#include "mujoco_ros_msgs/srv/set_body_state.hpp"
#include "mujoco_ros_msgs/srv/set_geom_properties.hpp"
#include "mujoco_ros_msgs/srv/set_equality_constraint_parameters.hpp"
#include "mujoco_ros_msgs/srv/get_equality_constraint_parameters.hpp"
#include "mujoco_ros_msgs/srv/get_geom_properties.hpp"
#include "mujoco_ros_msgs/srv/get_plugin_stats.hpp"
#include "mujoco_ros_msgs/srv/get_body_state.hpp"
#include "mujoco_ros_msgs/srv/get_state_uint.hpp"
#include "mujoco_ros_msgs/srv/get_gravity.hpp"
#include "mujoco_ros_msgs/srv/get_sim_info.hpp"

#include "mujoco_ros_msgs/action/step.hpp"


namespace mujoco_ros {

class MujocoEnvMutex : public std::recursive_mutex
{};
using MutexLock = std::unique_lock<std::recursive_mutex>;

struct CollisionFunctionDefault
{
	CollisionFunctionDefault(int geom_type1, int geom_type2, mjfCollision collision_cb)
	    : geom_type1_(geom_type1), geom_type2_(geom_type2), collision_cb_(collision_cb)
	{
	}

	int geom_type1_;
	int geom_type2_;
	mjfCollision collision_cb_;
};

struct OffscreenRenderContext
{
	mjvCamera cam;
	std::unique_ptr<unsigned char[]> rgb;
	std::unique_ptr<float[]> depth;
	std::shared_ptr<GLFWwindow> window;
	mjrContext con = {};
	mjvScene scn   = {};

	boost::thread render_thread_handle;

	// Condition variable to signal that the offscreen render thread should render a new frame
	std::atomic_bool request_pending = { false };

	std::mutex render_mutex;
	std::condition_variable_any cond_render_request;

	std::vector<rendering::OffscreenCameraPtr> cams;

	~OffscreenRenderContext();
};

class MujocoEnv
{
public:
	/**
	 * @brief Construct a new Mujoco Env object.
	 *
	 */
	MujocoEnv(std::shared_ptr<rclcpp::Node> node, const std::string &admin_hash = std::string());
	~MujocoEnv();

	MujocoEnv(const MujocoEnv &) = delete;

	// constants
	static constexpr int kErrorLength       = 1024;
	static constexpr int kMaxFilenameLength = 1000;

	const double syncMisalign       = 0.1; // maximum mis-alignment before re-sync (simulation seconds)
	const double simRefreshFraction = 0.7; // fraction of refresh available for simulation

	/// Noise to apply to control signal
	mjtNum *ctrlnoise_     = nullptr;
	double ctrl_noise_std  = 0.0;
	double ctrl_noise_rate = 0.0;

	mjvScene scn_;
	mjvPerturb pert_;

	MujocoEnvMutex physics_thread_mutex_;

	void connectViewer(Viewer *viewer);
	void disconnectViewer(Viewer *viewer);

	char queued_filename_[kMaxFilenameLength];

	struct
	{
		// Render options
		bool headless         = false;
		bool render_offscreen = false;
		bool use_sim_time     = true;

		// Sim speed
		int real_time_index = 8;
		int busywait        = 0;

		// Mode
		bool eval_mode = false;
		char admin_hash[64];

		// Atomics for multithread access
		std::atomic_int run                 = { 0 };
		std::atomic_int exit_request        = { 0 };
		std::atomic_int visual_init_request = { 0 };

		// Load request
		//  0: no request
		//  1: replace model_ with mnew and data_ with dnew
		//  2: load mnew and dnew from file
		std::atomic_int load_request      = { 0 };
		std::atomic_int reset_request     = { 0 };
		std::atomic_int speed_changed     = { 0 };
		std::atomic_int env_steps_request = { 0 };

		// Must be set to true before loading a new model from python
		std::atomic_int is_python_request = { 0 };
	} settings_;

	// General sim information for viewers to fetch
	struct
	{
		float measured_slowdown = 1.0;
		bool model_valid        = false;
		uint load_count         = 0;
	} sim_state_;

	std::vector<MujocoPluginPtr> const &getPlugins() const { return plugins_; }

	/**
	 * @brief Register a custom collision function for collisions between two geom types.
	 *
	 * @param [in] geom_type1 first geom type of the colliding geoms.
	 * @param [in] geom_type2 second type of the colliding geoms.
	 * @param [in] collision_cb collision function to call.
	 */
	void registerCollisionFunction(int geom_type1, int geom_type2, mjfCollision collision_cb);

	/**
	 * @brief Register a static transform to be published by the simulation.
	 *
	 * @param [in] transform const pointer to transform that will be published.
	 */
	void registerStaticTransform(geometry_msgs::msg::TransformStamped &transform);

	void waitForPhysicsJoin();
	void waitForEventsJoin();

	void startPhysicsLoop();
	void startEventLoop();

	/**
	 * @brief Get information about the current simulation state.
	 *
	 * Additionally to the `settings_.load_request` state, this function also considers visual initialization to be a
	 * part of the loading process.
	 *
	 * @return 0 if done loading, 1 if loading is in progress, 2 if loading has been requested.
	 */
	int getOperationalStatus();

	static constexpr float percentRealTime[] = {
		-1, // unbound
		2000, 1000, 800, 600,  500,  400, 200,  150,  100, 80,  66,   50,  40,  33,   25,   20,  16,   13,   10, 8,
		6.6f, 5.0f, 4,   3.3f, 2.5f, 2,   1.6f, 1.3f, 1,   .8f, .66f, .5f, .4f, .33f, .25f, .2f, .16f, .13f, .1f
	};

	static MujocoEnv *instance;
	static void proxyControlCB(const mjModel * /*m*/, mjData * /*d*/)
	{
		if (MujocoEnv::instance != nullptr)
			MujocoEnv::instance->runControlCbs();
	}
	static void proxyPassiveCB(const mjModel * /*m*/, mjData * /*d*/)
	{
		if (MujocoEnv::instance != nullptr)
			MujocoEnv::instance->runPassiveCbs();
	}

	// Proxies to MuJoCo callbacks
	void runControlCbs();
	void runPassiveCbs();

	bool togglePaused(bool paused, const std::string &admin_hash = std::string());

	GlfwAdapter *gui_adapter_ = nullptr;

	void runRenderCbs(mjvScene *scene);
	bool step(int num_steps = 1, bool blocking = true);

	void UpdateModelFlags(const mjOption *opt);

protected:
	std::vector<MujocoPlugin *> cb_ready_plugins_; // objects managed by plugins_
	XmlRpc::XmlRpcValue rpc_plugin_config_;
	std::vector<MujocoPluginPtr> plugins_;

	// This variable keeps track of remaining steps if the environment was configured to terminate after a fixed number
	// of steps (-1 means no limit).
	int num_steps_until_exit_ = -1;

	// VFS for loading models from strings
	mjVFS vfs_;

	// Currently loaded model
	char filename_[kMaxFilenameLength];
	// last error message
	char load_error_[kErrorLength];

	// Store default collision functions to restore on reload
	std::vector<CollisionFunctionDefault> defaultCollisionFunctions;

	// Keep track of overriden collisions to throw warnings
	std::set<std::pair<int, int>> custom_collisions_;

	// Keep track of static transforms to publish.
	std::vector<geometry_msgs::msg::TransformStamped> static_transforms_;

	// Central broadcaster for all static transforms
	tf2_ros::StaticTransformBroadcaster static_broadcaster_;

	// ROS TF2
	std::unique_ptr<tf2_ros::Buffer> tf_bufferPtr_;
	std::unique_ptr<tf2_ros::TransformListener> tf_listenerPtr_;

	/// Pointer to mjModel
	mjModelPtr model_; // technically could be a unique_ptr, but setting the deleter correctly is not trivial
	/// Pointer to mjData
	mjDataPtr data_; // technically could be a unique_ptr, but setting the deleter correctly is not trivial

	std::vector<Viewer *> connected_viewers_;

	void publishSimTime(mjtNum time);
	rclcpp::Publisher<rosgraph_msgs::msg::Clock>::SharedPtr clock_pub_;
	std::shared_ptr<rclcpp::Node> node_;

	void runLastStageCbs();

	void notifyGeomChanged(const int geom_id);

	template <typename T>
	std::vector<rclcpp::Service<T>::SharedPtr> service_servers_;
	std::unique_ptr<actionlib::SimpleActionServer<mujoco_ros_msgs::StepAction>> action_step_;

	bool verifyAdminHash(const std::string &hash);

	void setupServices();
	bool setPauseCB(const std::shared_ptr<mujoco_ros_msgs::srv::SetPause::Request> req, 
					std::shared_ptr<mujoco_ros_msgs::srv::SetPause::Response> resp);
	bool shutdownCB(const std::shared_ptr<std_srvs::srv::Empty::Request> req, 
					std::shared_ptr<std_srvs::srv::Empty::Response> resp);
	bool reloadCB(const std::shared_ptr<mujoco_ros_msgs::srv::Reload::Request> req, 
				  std::shared_ptr<mujoco_ros_msgs::srv::Reload::Response> resp);
	bool resetCB(const std::shared_ptr<std_srvs::srv::Empty::Request> req, 
				 std::shared_ptr<std_srvs::srv::Empty::Response> resp);
	bool setBodyStateCB(const std::shared_ptr<mujoco_ros_msgs::srv::SetBodyState::Request> req,
					    std::shared_ptr<mujoco_ros_msgs::srv::SetBodyState::Response> resp);
	bool getBodyStateCB(const std::shared_ptr<mujoco_ros_msgs::srv::GetBodyState::Request> req,
						std::shared_ptr<mujoco_ros_msgs::srv::GetBodyState::Response> resp);
	bool setGravityCB(const std::shared_ptr<mujoco_ros_msgs::srv::SetGravity::Request> req, 
					  std::shared_ptr<mujoco_ros_msgs::srv::SetGravity::Response> resp);
	bool getGravityCB(const std::shared_ptr<mujoco_ros_msgs::srv::GetGravity::Request> req, 
					  std::shared_ptr<mujoco_ros_msgs::srv::GetGravity::Response> resp);
	bool setGeomPropertiesCB(const std::shared_ptr<mujoco_ros_msgs::srv::SetGeomProperties::Request> req,
	                         std::shared_ptr<mujoco_ros_msgs::srv::SetGeomProperties::Response> resp);
	bool getGeomPropertiesCB(const std::shared_ptr<mujoco_ros_msgs::srv::GetGeomProperties::Request> req,
	                         std::shared_ptr<mujoco_ros_msgs::srv::GetGeomProperties::Response> resp);
	bool setEqualityConstraintParametersArrayCB(const std::shared_ptr<mujoco_ros_msgs::srv::SetEqualityConstraintParameters::Request> req,
	                                            std::shared_ptr<mujoco_ros_msgs::srv::SetEqualityConstraintParameters::Response> resp);
	bool getEqualityConstraintParametersArrayCB(const std::shared_ptr<mujoco_ros_msgs::srv::GetEqualityConstraintParameters::Request> req,
	                                            std::shared_ptr<mujoco_ros_msgs::srv::GetEqualityConstraintParameters::Response> resp);
	bool getStateUintCB(const std::shared_ptr<mujoco_ros_msgs::srv::GetStateUint::Request> req, 
					    std::shared_ptr<mujoco_ros_msgs::srv::GetStateUint::Response> resp);
	bool getSimInfoCB(const std::shared_ptr<mujoco_ros_msgs::srv::GetSimInfo::Request> req, 
					  std::shared_ptr<mujoco_ros_msgs::srv::GetSimInfo::Response> resp);
	bool setRTFactorCB(const std::shared_ptr<mujoco_ros_msgs::srv::SetFloat::Request> req, 
					   std::shared_ptr<mujoco_ros_msgs::srv::SetFloat::Response> resp);
	bool getPluginStatsCB(const std::shared_ptr<mujoco_ros_msgs::srv::GetPluginStats::Request> req,
	                      std::shared_ptr<mujoco_ros_msgs::srv::GetPluginStats::Response> resp);
	bool setEqualityConstraintParameters(const mujoco_ros_msgs::msg::EqualityConstraintParameters &parameters);
	bool getEqualityConstraintParameters(mujoco_ros_msgs::msg::EqualityConstraintParameters &parameters);

	// Action calls
	void onStepGoal(const mujoco_ros_msgs::StepGoalConstPtr &goal);

	void resetSim();

	/**
	 * @brief Loads and sets the initial joint states from the parameter server.
	 */
	void loadInitialJointStates();

	void setJointPosition(const double &pos, const int &joint_id, const int &jnt_axis /*= 0*/);
	void setJointVelocity(const double &vel, const int &joint_id, const int &jnt_axis /*= 0*/);

	/**
	 * @brief Makes sure that all data that will be replaced in a reload is freed.
	 */
	void prepareReload();

	// Threading

	boost::thread physics_thread_handle_;
	boost::thread event_thread_handle_;

	// Helper variables to get the state of threads
	std::atomic_int is_physics_running_   = { 0 };
	std::atomic_int is_event_running_     = { 0 };
	std::atomic_int is_rendering_running_ = { 0 };

	/**
	 * @brief Runs physics steps.
	 */
	void physicsLoop();

	/**
	 * @brief physics step when sim is running.
	 */
	void simUnpausedPhysics(mjtNum &syncSim, std::chrono::time_point<Clock> &syncCPU);

	/**
	 * @brief physics step when sim is paused.
	 */
	void simPausedPhysics(mjtNum &syncSim);

	/**
	 * @brief Handles requests from other threads (viewers).
	 */
	void eventLoop();

	void completeEnvSetup();

	/**
	 * @brief Tries to load all configured plugins.
	 * This function is called when a new mjData object is assigned to the environment.
	 */
	void loadPlugins();

	void initializeRenderResources();

	OffscreenRenderContext offscreen_;

	void offscreenRenderLoop();

	// Model loading
	mjModel *mnew = nullptr;
	mjData *dnew  = nullptr;

	/**
	 * @brief Load a queued model from either a path or XML-string.
	 */
	bool initModelFromQueue();

	/**
	 * @brief Replace the current model and data with new ones and complete the loading process.
	 */
	void loadWithModelAndData();

	mjThreadPool *threadpool_ = nullptr;
};

} // end namespace mujoco_ros
