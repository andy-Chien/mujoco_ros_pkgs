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
#include <rclcpp/rclcpp.hpp>
#include <mujoco_ros/glfw_adapter.h>
#include <mujoco_ros/viewer.h>
#include <mujoco_ros/mujoco_env.h>

#include <mujoco_ros/array_safety.h>
#include <mujoco/mujoco.h>

#include <boost/program_options.hpp>
#include <csignal>
#include <thread>

namespace {
namespace po  = boost::program_options;
namespace mju = ::mujoco::sample_util;

using Seconds = std::chrono::duration<double>;
} // anonymous namespace

class MujocoRos : public rclcpp::Node
{
public:
    MujocoRos(){
        declare_parameters();
        read_parameters();
        init();
    }
    ~MujocoRos(){
        env->waitForPhysicsJoin();
        env->waitForEventsJoin();
        env.reset();
        RCLCPP_INFO(this->get_logger(), "MuJoCo ROS Simulation Server node is terminating");
    }
private:
    void sigint_handler(int /*sig*/)
    {
        std::printf("Registered C-c. Shutting down MuJoCo ROS Server ...\n");
        env->settings_.exit_request.store(1);
    }

    void declare_parameters(){
        this->declare_parameter<bool>("no_x", false);
        this->declare_parameter<bool>("headless", false);
        this->declare_parameter<bool>("wait_for_xml", false);
        this->declare_parameter<std::string>("modelfile", "");
        this->declare_parameter<std::string>("mujoco_xml", "");
    }

    void read_parameters(){
        /*
        * Model (file) passing: the model can be provided as file to parse or directly as string stored in the rosparam
        * server. If both string and file are provided, the string takes precedence.
        */
       
        modelfile = this->get_parameter("modelfile").as_string();
        wait_for_xml = this->get_parameter("wait_for_xml").as_bool();
        xml_content_path = this->get_parameter("mujoco_xml").as_string();
        no_x = this->get_parameter("no_x").as_bool() || this->get_parameter("headless").as_bool();

        RCLCPP_INFO_EXPRESSION(this->get_logger(), wait_for_xml, "Waiting for xml content to be available on rosparam server");

        if (wait_for_xml && !xml_content_path.empty()) {
            this->declare_parameter<std::string>(xml_content_path, "");
            xml_content = this->get_parameter(xml_content_path).as_string();
            if (!xml_content.empty()) {
                RCLCPP_INFO(this->get_logger(), "Got xml content from ros param server");
                modelfile = "rosparam_content";
            }
            wait_for_xml = false;
        }

        if (!modelfile.empty()) {
            RCLCPP_INFO_STREAM(this->get_logger(), "Using modelfile " << modelfile);
        } else {
            RCLCPP_WARN(this->get_logger(), "No modelfile was provided, launching empty simulation!");
        }
    }

    void init()
    {
        signal(SIGINT, sigint_handler);

        std::string admin_hash("");

        po::options_description options;
        options.add_options() // clang-format off
        ("help,h", "Produce this help message")
        ("admin-hash", po::value<std::string>(&admin_hash),"Set the admin hash for eval mode.");
        // clang-format on
        po::variables_map vm;

        try {
            po::store(po::parse_command_line(argc, argv, options), vm);
            po::notify(vm);

            if (vm.count("help")) {
                std::cout << "command line options:\n" << options;
                exit(0);
            }
        } catch (std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Error parsing command line: %s", e.what());
            exit(-1);
        }
    
        std::printf("MuJoCo version %s\n", mj_versionString());
        if (mjVERSION_HEADER != mj_version()) {
            RCLCPP_ERROR_STREAM(this->get_logger(), "Headers (" << mjVERSION_HEADER << ") and library (" << mj_versionString()
                                        << ") have different versions");
            mju_error("Headers and library have different versions");
        }

        // TODO(dleins): Should MuJoCo Plugins be loaded?
        env = std::make_unique<mujoco_ros::MujocoEnv>(admin_hash);

        // const char *modelfile = nullptr;
        if (!modelfile.empty()) {
            mju::strcpy_arr(env->queued_filename_, modelfile.c_str());
            env->settings_.load_request = 2;
        }

        env->startPhysicsLoop();
        env->startEventLoop();

        if (!no_x) {
            // mjvCamera cam;
            // mjvOption opt;
            // mjv_defaultCamera(&cam);
            // mjv_defaultOption(&opt);
            RCLCPP_INFO(this->get_logger(), "Launching viewer");
            viewer =
                // std::make_unique<mujoco_ros::Viewer>(std::unique_ptr<mujoco_ros::PlatformUIAdapter>(env->gui_adapter_),
                //                                      env.get(), &cam, &opt, /* is_passive = */ false);
                std::make_unique<mujoco_ros::Viewer>(std::unique_ptr<mujoco_ros::PlatformUIAdapter>(env->gui_adapter_),
                                                    env.get(), /* is_passive = */ false);
            viewer->RenderLoop();
        } else {
            RCLCPP_INFO(this->get_logger(), "Running headless");
        }
    }

    bool no_x, wait_for_xml;
    std::string modelfile, xml_content_path, xml_content;
    std::unique_ptr<mujoco_ros::MujocoEnv> env;
    std::unique_ptr<mujoco_ros::Viewer> viewer;
}
