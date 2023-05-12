 /*
 * @name	 	triangle_profile.h
 * @brief	 	Simple triangle walking pattern.
 * @author  Ronja Gueldenring
 * @date 		2019/04/05
 **/

#include <ros/ros.h>
#include <arena_plugins/walking_profile.h>


#ifndef TRIANGLE_PROFILE_H
#define TRIANGLE_PROFILE_H



namespace flatland_plugins {

/**
 * @class WalkingProfile
 * @brief Realizes a simple triangular walking profile.
 * 
 */
class TriangleProfile : public WalkingProfile {
 public:
    TriangleProfile(double step_time);
     /**
     * @brief Computes the speed multiplier of body_speed.
     * body_speed*speed_multiplier is speed of the leg
     */
    double get_speed_multiplier(double body_speed);

    /**
     * @brief Returns true if leg is in state CENTER of swing phase (maximum speed)
     */
    bool is_leg_in_center();

  private:
    enum States
    {
        INIT,
        SWING_PHASE1,
        CENTER,
        SWING_PHASE2,
    };

    double step_time_;            // time a step should approximately last
    double start_time_;           // time when step starts
    int state_;                   // swing state of leg

    /**
     * @brief Calls State Machine.
     */
    void nextState();            

};
};

#endif /* TRIANGLE_PROFILE_H */
