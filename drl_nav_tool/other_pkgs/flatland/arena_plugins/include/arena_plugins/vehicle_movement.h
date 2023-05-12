 /*
 * @name	 	pedsim_movement.cpp
 * @brief	 	The movement of the pedsim agents is as well applied to the flatland models.
 *              Furthermore, a walking pattern is added.
 * @author  	Ronja Gueldenring
 * @date 		2019/04/05
 **/

#include <flatland_plugins/update_timer.h>
#include <flatland_server/model_plugin.h>
#include <flatland_server/timekeeper.h>
#include <flatland_server/body.h>
#include <flatland_server/types.h>
#include <pedsim_msgs/AgentStates.h>
#include <pedsim_msgs/AgentState.h>
#include <ros/ros.h>
#include <flatland_msgs/DangerZone.h>
#include <tf/transform_listener.h>
#include <arena_plugins/triangle_profile.h>
#include<cmath> 
#include <ros/package.h>

#ifndef FLATLAND_PLUGINS_VEHICLE_MOVEMENT_H
#define FLATLAND_PLUGINS_VEHICLE_MOVEMENT_H

using namespace flatland_server;

namespace flatland_plugins {

/**
 * @class VehicleMovement
 * @brief The movement of the pedsim agents is as well applied to the flatland models.
 */
class VehicleMovement : public ModelPlugin {
 public:
  /**
   * @brief Initialization for the plugin
   * @param[in] config Plugin YAML Node
   */
  void OnInitialize(const YAML::Node &config) override;

  /**
   * @brief Called when just before physics update
   * @param[in] timekeeper Object managing the simulation time
   */
  void BeforePhysicsStep(const Timekeeper &timekeeper) override;

   /**
    * @name          AfterPhysicsStep
    * @brief         override the AfterPhysicsStep method
    * @param[in] timekeeper Object managing the simulation time
    */
  void AfterPhysicsStep(const Timekeeper& timekeeper) override;


  private: 
    b2Body * body_;                            ///< Pointer to base-body
    UpdateTimer update_timer_;              ///< for controlling update rate

    b2Body * safety_dist_b2body_;               ///< Pointer to safety distance circle
    Body * safety_dist_body_;               ///< Pointer to safety distance circle
    pedsim_msgs::AgentState person;
    ros::Subscriber pedsim_agents_sub_;        ///< Subscriber to pedsim agents state
    ros::Publisher agent_state_pub_;          ///< Publisher for agent state of  every pedsim agent
    
    tf::TransformListener listener_;           ///< Transform Listner
    pedsim_msgs::AgentStatesConstPtr agents_;  ///< most recent pedsim agent state
   
    double safety_dist_;
    double safety_dist_original_;
    std::string body_frame_;                   ///< frame name of base-body
    
    bool useDangerZone;
    float vel_x;
    float vel_y;
    float vel;
    float human_radius;
    float dangerZoneRadius;
    float dangerZoneAngle;                                    //dangerZoneAngle
    std::vector<float> pL;                                         //dangerZoneCenter in the agent frame             
    std::vector<double> dangerZoneCenter; //dangerZoneCenter in absolute frame  
    ros::Publisher danger_zone_pub_; 
    flatland_msgs::DangerZone dangerZone;
    
    flatland_plugins::TriangleProfile* wp_;
    Color c;
    float safety_dist_body_alpha;


    //parameters for calculating danger zone
    float slopeBE1;
    float slopeBE2;
    float mv;
    float av;
    float r_static;
    // std::vector<float> pA
    float pB_1;
    float pB_2;
    // std::vector<float> pC;
    float a;
    float b; 
    float c_; 
    float h;
    float interceptBE1;
    float interceptBE2;
    std::vector<float> interceptBE;
    std::vector<float> slopeBE;
    std::vector<double> velocityAngles;

   
    /**
     * @brief Callback for pedsim agent topic
     * @param[in] agents array of all agents
     */
    void agentCallback(const pedsim_msgs::AgentStatesConstPtr& agents);

    /**
     * @brief Method is copy of model_body.cpp to be able to change radius programatically
     * ToDo: Find more elegeant solution!
     */
    void set_safety_dist_footprint(b2Body * physics_body_, double radius);

    /**
     * @brief To be able to change radius programatically
     */
    void ConfigFootprintDefSafetyDist(b2FixtureDef &fixture_def); 

    /**
     * @brief update safety distance circle, when the agent is chatting.
     * Body Footprint of safety dist circle will be set.
     */
    void updateSafetyDistance();

        /**
   * @brief visualize the dangerous zone according to the velocities of human.
   * Body Footprints of dangrous zones will be set.
   */
    void updateDangerousZone(float p, float radius, float angle);

  /**
   * @brief calculate the dangerous zone
   * */
    void calculateDangerZone(float vel);
    
};
};

#endif
