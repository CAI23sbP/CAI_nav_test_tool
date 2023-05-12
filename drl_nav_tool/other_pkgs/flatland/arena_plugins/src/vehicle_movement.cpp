 /*
 * @name	 	vehicle_movement.cpp
 * @brief	 	The movement of the pedsim agents is as well applied to the flatland models.
 **/

#include <arena_plugins/vehicle_movement.h>
#include <arena_plugins/triangle_profile.h>
#include <flatland_server/exceptions.h>
#include <flatland_server/yaml_reader.h>
#include <pluginlib/class_list_macros.h>
#include<bits/stdc++.h>
#include <iostream>
#include <pwd.h>
#include <string>
using namespace flatland_server;

namespace flatland_plugins {

void VehicleMovement::OnInitialize(const YAML::Node &config){
    agents_ = NULL;

    //get parameters
    flatland_server::YamlReader reader(config);
    std::string path = ros::package::getPath("simulation-setup");
    YAML::Node config1 = YAML::LoadFile(path+"/advanced_configs.yaml");
    human_radius=0.4;
    mv = 1.5;
    av =1.5;
    r_static = 0.7;
    useDangerZone=false;
    if (config1["use danger zone"].as<float>()== 1.0) 
    {useDangerZone=true;}

    safety_dist_ = reader.Get<double>("safety_dist");
    safety_dist_original_ = safety_dist_;
    //Subscribing to pedsim topic to apply same movement
    std::string pedsim_agents_topic = ros::this_node::getNamespace() + reader.Get<std::string>("agent_topic");
    
    std::string agent_state_topic = reader.Get<std::string>("agent_state_pub", "agent_state");

    // Subscribe to ped_sims agent topic to retrieve the agents position
    pedsim_agents_sub_ = nh_.subscribe(pedsim_agents_topic, 1, &VehicleMovement::agentCallback, this);
    // publish pedsim agent AgentState
    agent_state_pub_ = nh_.advertise<pedsim_msgs::AgentState>(agent_state_topic, 1);

    //Get bodies of pedestrian
    body_ = GetModel()->GetBody(reader.Get<std::string>("base_body"))->GetPhysicsBody();
    safety_dist_b2body_ = GetModel()->GetBody(reader.Get<std::string>("safety_dist_body"))->GetPhysicsBody();
    safety_dist_body_ = GetModel()->GetBody(reader.Get<std::string>("safety_dist_body"));
    updateSafetyDistance();

    // check if valid bodies are given
    if (body_ == nullptr) {
        throw flatland_server::YAMLException("Body with with the given name does not exist");
    }

    safety_dist_body_alpha = reader.Get<float>("safety_dist_body_alpha", 0.3);
}

void VehicleMovement::updateSafetyDistance(){
    set_safety_dist_footprint(safety_dist_b2body_, safety_dist_);
}

void VehicleMovement::BeforePhysicsStep(const Timekeeper &timekeeper) {
    if (agents_ == NULL) {
        return;
    }
      
    
    std::string path = ros::package::getPath("simulation-setup");
    YAML::Node config = YAML::LoadFile(path+"/saftey_distance_parameter_none.yaml");
    // get agents ID via namespace
    std::string ns_str = GetModel()->GetNameSpace();
    int id_ = std::stoi(ns_str.substr(13, ns_str.length()));


    //Find appropriate agent in list
    for (int i = 0; i < (int) agents_->agent_states.size(); i++){
        pedsim_msgs::AgentState p = agents_->agent_states[i];
        if (p.id == id_){
            person = p;
            break;
        }
        if (i == agents_->agent_states.size() - 1) {
            ROS_WARN("Couldn't find agent: %d", id_);
            return;
        }
    };
    //modeling of safety distance
    vel_x = person.twist.linear.x; //
    vel_y = person.twist.linear.y; // 
    vel = sqrt(vel_x*vel_x+vel_y*vel_y);
    //change visualization of the human if they are talking
    safety_dist_= config["safety distance factor"][person.social_state].as<float>() * config["human obstacle safety distance radius"][person.type].as<float>();

    c=Color(  0.26, 0.3, 0, safety_dist_body_alpha) ;
    if ( config["safety distance factor"][person.social_state].as<float>() > 1.2  ){
            c=Color(0.93, 0.16, 0.16, safety_dist_body_alpha);
    }
    else if(config["safety distance factor"][person.social_state].as<float>() < 0.89){  
            c=Color(  0.16, 0.93, 0.16, safety_dist_body_alpha) ;
    }
    if(useDangerZone==false){
            //change visualization of the human if they are talking         
      
            safety_dist_body_->SetColor(c);
            updateSafetyDistance();
    }else{
        dangerZoneCenter.clear();
        if(vel>0.01){ //this threshold is used for filtering of some rare cases, no influence for performance
            calculateDangerZone(vel);
            velocityAngles.clear();
            double velocityAngle=atan(vel_y/vel_x);
            velocityAngles.push_back(velocityAngle);
            velocityAngles.push_back(velocityAngle+M_PI);
            for(int i=0; i<2; i++){
                double x=pL[0]*cos(velocityAngles[i]);
                double y=pL[0]*sin(velocityAngles[i]);
                if(x*vel_x+y*vel_y<0){
                    dangerZoneCenter.push_back(person.pose.position.x+x);
                    dangerZoneCenter.push_back(person.pose.position.y+y);
                    break;
                }
            }
        }else{// if vel <0.01, it is treated as stopped
            dangerZoneRadius=safety_dist_original_;
            dangerZoneAngle=2*M_PI;        
            dangerZoneCenter.push_back(person.pose.position.x);
            dangerZoneCenter.push_back(person.pose.position.y);
        }
        //
        dangerZone.header=person.header;
        dangerZone.dangerZoneRadius=dangerZoneRadius;
        dangerZone.dangerZoneAngle=dangerZoneAngle;
        dangerZone.dangerZoneCenter=dangerZoneCenter;
    }
 
    float vel_x = person.twist.linear.x;
    float vel_y = person.twist.linear.y;
    float angle_soll = person.direction;
    float angle_ist = body_->GetAngle();

    //Set pedsim_agent position in flatland simulator
    body_->SetTransform(b2Vec2(person.pose.position.x, person.pose.position.y), angle_soll);
    safety_dist_b2body_->SetTransform(b2Vec2(person.pose.position.x, person.pose.position.y), angle_soll);
    
    //Set pedsim_agent velocity in flatland simulator to approach next position
    body_->SetLinearVelocity(b2Vec2(vel_x, vel_y));
    safety_dist_b2body_->SetLinearVelocity(b2Vec2(vel_x, vel_y));
}

// ToDo: Implelent that more elegant
// Copied this function from model_body.cpp in flatland folder
// This is necessary to be able to set the leg radius auto-generated with variance
// original function just applies the defined radius in yaml-file.
// other option: modify flatland package, but third-party
void VehicleMovement::set_safety_dist_footprint(b2Body * physics_body, double radius){
    Vec2 center = Vec2(0, 0);
    b2FixtureDef fixture_def;
    ConfigFootprintDefSafetyDist(fixture_def);

    b2CircleShape shape;
    shape.m_p.Set(center.x, center.y);
    shape.m_radius = radius;

    fixture_def.shape = &shape;
    b2Fixture* old_fix = physics_body->GetFixtureList();
    physics_body->DestroyFixture(old_fix);
    physics_body->CreateFixture(&fixture_def);
}

// ToDo: Implelent that more elegant
// Copied this function from model_body.cpp in flatland folder
// This is necessary to be able to set the leg radius auto-generated with variance
// original function just applies the defined properties from yaml-file.
// other option: modify flatland package, but third-party
void VehicleMovement::ConfigFootprintDefSafetyDist(b2FixtureDef &fixture_def) {
    // configure physics properties
    fixture_def.density = 0.0;
    fixture_def.friction = 0.0;
    fixture_def.restitution = 0.0;

    // config collision properties
    fixture_def.isSensor = true;
    fixture_def.filter.groupIndex = 0;

    // Defines that body is just seen in layer "2D" and "ped"
    fixture_def.filter.categoryBits = 0x000a;

    bool collision = false;
    if (collision) {
        // b2d docs: maskBits are "I collide with" bitmask
        fixture_def.filter.maskBits = fixture_def.filter.categoryBits;
    } else {
        // "I will collide with nothing"
        fixture_def.filter.maskBits = 0;
    }
}

void VehicleMovement::agentCallback(const pedsim_msgs::AgentStatesConstPtr& agents){
    agents_ = agents;
}

void VehicleMovement::AfterPhysicsStep(const Timekeeper& timekeeper) {
  bool publish = update_timer_.CheckUpdate(timekeeper);
  if (publish) {
    // get the state of the body and publish the data
    // publish agent state for every human
    //publish the agent state 
    agent_state_pub_.publish(person);
  }
}



void VehicleMovement::calculateDangerZone(float vel_agent){
    interceptBE.clear();
    slopeBE.clear();
    pL.clear();
    dangerZoneRadius = mv*vel_agent + r_static;
    dangerZoneAngle = 11*M_PI / 6* exp(-1.4*av*vel_agent) +  M_PI/6;
    pB_1 = dangerZoneRadius*cos(dangerZoneAngle/2);
    pB_2 = dangerZoneRadius*sin(dangerZoneAngle/2);
    // pC_1 = dangerZoneRadius*cos(- dangerZoneAngle/2);
    // pC_2= dangerZoneRadius*sin(- dangerZoneAngle/2;
    // float diffY=-pB[1];
    // float diffX=-pB[0];
    a = human_radius*human_radius - pB_1*pB_1;
    b = 2*pB_1*pB_2;
    c_ = human_radius*human_radius - pB_2*pB_2;
    h = b*b - 4*a*c_;
    if(h<0){
        ROS_INFO("no valid root for m+++++h=[%f]",h);
    }else{
        slopeBE1 = (-b+sqrt(h))/(2*a);
        slopeBE2 = (-b-sqrt(h))/(2*a);
    }
    interceptBE1 = pB_2 - slopeBE1*pB_1;
    interceptBE2 = pB_2 - slopeBE2*pB_1;
    interceptBE.push_back(interceptBE1);
    interceptBE.push_back(interceptBE2);
    slopeBE.push_back(slopeBE1);
    slopeBE.push_back(slopeBE2);
    for(int i= 0; i< 2; i++)
    {    
        float x = (- interceptBE[i])/(slopeBE[i]);
        float y = slopeBE[i]*x + interceptBE[i];
        float vAEx=x;
        float vAEy=y;
        if(vAEx*vel_agent<0){
            pL.push_back(x);
            pL.push_back(y);
            break;
        }
    }
    float vLBx=pL[0]-pB_1;
    float vLBy=pL[1]-pB_2;
    float vLAx=pL[0];
    float vLAy=pL[1];
    float dotProductLBLA =vLBx*vLAx+vLBy*vLAy;
    float normLB = sqrt(vLBx*vLBx+vLBy*vLBy);
    float normLA = sqrt(vLAx*vLAx+vLAy*vLAy);
    if (dangerZoneAngle < M_PI){
        float c1 = dotProductLBLA/(normLB*normLA);
        //clamp(-1,1)
        float c=c1>1 ? 1 : c1;
        c=c1<-1? -1 :c1;
        float angle = acos(c);
        dangerZoneAngle = 2*angle;
    }
    // ROS_INFO("safty model pE0[%f]dangerZoneRadius[%f]dangerZoneAngle[%f]", pL[0], dangerZoneRadius, dangerZoneAngle);
    updateDangerousZone(pL[0], dangerZoneRadius, dangerZoneAngle);
}


void VehicleMovement::updateDangerousZone(float p, float radius, float angle){
    //destroy the old fixtures 
    for(int i = 0; i<12; i++){
        b2Fixture* old_fix = safety_dist_b2body_->GetFixtureList();
        if(old_fix==nullptr){break;}
        safety_dist_b2body_->DestroyFixture(old_fix);
    }
    // create new feature
    b2FixtureDef fixture_def;
    // configure physics properties
    fixture_def.density = 1.0;
    fixture_def.friction = 0.0;
    fixture_def.restitution = 0.0;
    // config collision properties
    fixture_def.isSensor = true;
    fixture_def.filter.groupIndex = 0;
    // Defines that body is just seen in layer "2D" and "ped"
    fixture_def.filter.categoryBits = 0x000a;
    bool collision = false;
    if (collision) {
        // b2d docs: maskBits are "I collide with" bitmask
        fixture_def.filter.maskBits = fixture_def.filter.categoryBits;
    } else {
        // "I will collide with nothing"
        fixture_def.filter.maskBits = 0;
    }
    float delta = angle/10;
    float last_angle = -angle/2;
    float next_angle;
    float v1 = radius*cos(last_angle);
    float v2 = radius*sin(last_angle);
    b2PolygonShape shape;
    b2Vec2 verts[3];
    for(int i = 0; i<10; i++){
        next_angle= -angle/2 + (i+1)*delta;
        verts[0].Set(0.0, 0.0);
        verts[1].Set(v1, v2);
        v1= radius*cos(next_angle);
        v2= radius*sin(next_angle);
        verts[2].Set(v1, v2);
        shape.Set(verts, 3);
        fixture_def.shape = &shape;
        safety_dist_b2body_->CreateFixture(&fixture_def);
    }    
    if(p==0.0){
        verts[0].Set(0.0, 0.0);
        v1= radius*cos(-angle/2);
        v2= radius*sin(-angle/2);
        verts[1].Set(v1, v2);
        v1= radius*cos(angle/2);
        v2= radius*sin(angle/2);
        verts[2].Set(v1, v2);
        shape.Set(verts, 3);
        fixture_def.shape = &shape;
        safety_dist_b2body_->CreateFixture(&fixture_def);
    }else{
        //first vertex
        verts[0].Set(0.0, 0.0);
        verts[1].Set(p, 0.0);
        v1= radius*cos(angle/2);
        v2= radius*sin(angle/2);
        verts[2].Set(v1, v2);
        shape.Set(verts, 3);
        fixture_def.shape = &shape;
        safety_dist_b2body_->CreateFixture(&fixture_def);
        //second vertex
        verts[0].Set(0.0, 0.0);
        verts[1].Set(p, 0.0);
        v1= radius*cos(-angle/2);
        v2= radius*sin(-angle/2);
        verts[2].Set(v1, v2);
        shape.Set(verts, 3);
        fixture_def.shape = &shape;
        safety_dist_b2body_->CreateFixture(&fixture_def);
    }
    safety_dist_body_->SetColor(c);
}

};

PLUGINLIB_EXPORT_CLASS(flatland_plugins::VehicleMovement, flatland_server::ModelPlugin)

