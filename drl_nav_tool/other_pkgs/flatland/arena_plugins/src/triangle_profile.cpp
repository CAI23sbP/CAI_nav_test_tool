 /*
 * @name	 	triangle_profile.cpp
 * @brief	 	Simple triangle walking pattern.
 * @author  	Ronja Gueldenring
 * @date 		2019/04/05
 **/
#include <arena_plugins/triangle_profile.h>

namespace flatland_plugins {

    TriangleProfile::TriangleProfile(double step_time) : step_time_{step_time}{
        state_ = INIT;
        // step_time_=step_time_/7*5;
    }

    
    double TriangleProfile::get_speed_multiplier(double body_speed){
        double now = ros::Time::now().toSec();
        double max_vel = 4 * body_speed;
        double m = max_vel / (step_time_ * 0.5);
        double leg_vel = -max_vel;
        double diff = now - start_time_;
        
        switch (state_){
            case INIT:
                start_time_ = now - (step_time_ * 0.5);
                nextState();
                break;

            case SWING_PHASE1:
                leg_vel = m * diff;
                if(diff >= step_time_ * 0.5){
                    nextState();
                }
                break;

            case CENTER:
                leg_vel = max_vel - m * (diff-step_time_ * 0.5);
                nextState();
                break;

            case SWING_PHASE2:
                leg_vel = max_vel - m *(diff-step_time_ * 0.5);
                if (diff >= step_time_){
                    leg_vel = 0.0;
                    start_time_ = now;
                    nextState();
                }
                break;
        }

        if (body_speed < 0.01) {
            return 1.0;
        }

        return leg_vel/body_speed;
    }

    void TriangleProfile::nextState(){
        switch (state_){
            case INIT:
                state_ = SWING_PHASE2;
                break;

            case SWING_PHASE1:
                state_ = CENTER;
                break;

            case CENTER:
                state_ = SWING_PHASE2;
                break;

            case SWING_PHASE2:
                state_ = SWING_PHASE1;
                break;  
        }
    }

    bool TriangleProfile::is_leg_in_center(){
        return state_==CENTER;
    }
};

