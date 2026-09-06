import time
from navigation.schemas import AgentNavigationCommand, RobotCommand, SimulatorObservation

# Hardcoded safety rules for our prototype (from the Integration Spec)
MAX_ACTION_DURATION_S = 1.0
MIN_SAFE_OBSTACLE_DISTANCE_M = 0.5 # Half a meter
MAX_STALE_OBSERVATION_MS = 2000    # 2 seconds

class SafetySupervisor:
    """
    It checks real-world sensors and returns a safe RobotCommand.
    """
    
    def evaluate_command(self, agent_cmd: AgentNavigationCommand, obs: SimulatorObservation) -> RobotCommand:
        safe_action = "STOP"
        safe_vx = 0.0
        safe_duration = 0.1
        
        # Rule 1: Check if the robot is already in a collision or error state
        if obs.robot_status in ["COLLISION", "ERROR"]:
            print("[Safety] Robot is in error state. Forcing STOP.")
            return self._create_stop_command(agent_cmd)
            
        # Rule 2: Check for stale data (did we lose the WiFi connection to the robot?)
        current_time_ms = int(time.time() * 1000)
        time_since_obs_ms = current_time_ms - obs.timestamp_ms
        if time_since_obs_ms > MAX_STALE_OBSERVATION_MS:
            print(f"[Safety] Observation is stale ({time_since_obs_ms}ms old). Forcing STOP.")
            return self._create_stop_command(agent_cmd)
            
        # Rule 3: Check for obstacles blocking the path
        for obstacle in obs.obstacles:
            # We check both 'blocking' and 'is_blocking' just in case the simulator changes slightly
            if obstacle.get("blocking", False) or obstacle.get("is_blocking", False):
                if obstacle.get("distance_m", 999.0) < MIN_SAFE_OBSTACLE_DISTANCE_M:
                    print("[Safety] Obstacle too close! Forcing STOP.")
                    return self._create_stop_command(agent_cmd)
                    
        # Rule 4: If it is safe, clamp the duration so the robot doesn't drive forever!
        safe_action = self._map_agent_action_to_robot_action(agent_cmd.action)
        safe_duration = min(agent_cmd.max_duration_s, MAX_ACTION_DURATION_S)
        
        # For our simple prototype, MOVE just goes forward safely at 0.25 meters per second
        if safe_action == "MOVE":
            safe_vx = 0.25 
        
        # Finally, build the safe command to send to the motors
        return RobotCommand(
            run_id=agent_cmd.run_id,
            command_id=agent_cmd.command_id,
            action=safe_action,
            vx_mps=safe_vx,
            vy_mps=0.0,
            wz_rps=0.0,
            duration_s=safe_duration,
            target_waypoint=agent_cmd.waypoint_id,
            safety_checked=True
        )

    def _create_stop_command(self, agent_cmd: AgentNavigationCommand) -> RobotCommand:
        """Helper to quickly hit the brakes."""
        return RobotCommand(
            run_id=agent_cmd.run_id,
            command_id=agent_cmd.command_id,
            action="STOP",
            vx_mps=0.0, vy_mps=0.0, wz_rps=0.0,
            duration_s=0.1,
            target_waypoint=agent_cmd.waypoint_id,
            safety_checked=True
        )
        
    def _map_agent_action_to_robot_action(self, agent_action: str) -> str:
        """Translates the high-level intent to a low-level motor action."""
        if agent_action == "MOVE_TO_WAYPOINT":
            return "MOVE"
        elif agent_action in ["TURN_LEFT", "TURN_RIGHT"]:
            return "TURN"
        else:
            return "STOP" # YIELD, STOP, ASK_CLARIFICATION all mean the wheels stop moving