from src.navigation.schemas import AgentNavigationCommand, SimulatorObservation, NavigationFeedback

# If the robot is within 0.5 meters of the target, we consider it "arrived" at the waypoint.
WAYPOINT_REACHED_THRESHOLD_M = 0.5 

class RouteEvaluator:
    """
    Watches the robot's progress and tells the AI Agent 
    what happened (e.g. 'You reached the waypoint!' or 'A cart is blocking us!').
    """
    
    def evaluate_progress(self, agent_cmd: AgentNavigationCommand, obs: SimulatorObservation, was_stopped_by_safety: bool) -> NavigationFeedback:
        
        # 1. Did the Safety Supervisor hit the brakes because of an obstacle?
        if was_stopped_by_safety:
            for obstacle in obs.obstacles:
                if obstacle.get("blocking", False) or obstacle.get("is_blocking", False):
                    # We are blocked! Tell the Agent to replan the route.
                    return self._create_feedback(
                        agent_cmd, 
                        status="BLOCKED", 
                        obs=obs,
                        replan=True,
                        reason="Safety supervisor stopped robot due to blocking obstacle."
                    )
            
            # If we were stopped, but it wasn't an obstacle, it must be stale data or a crash.
            return self._create_feedback(
                agent_cmd, 
                status="STOPPED", 
                obs=obs,
                human_help=True,
                reason="Safety supervisor stopped robot due to unsafe conditions (e.g., stale camera)."
            )
            
        # 2. Did we reach the waypoint?
        distance = obs.distance_to_target_m
        if distance is not None and distance <= WAYPOINT_REACHED_THRESHOLD_M:
            
            # Was this the very last waypoint of the whole trip? (Zero-indexed, so we subtract 1)
            if agent_cmd.waypoint_index >= (agent_cmd.total_waypoints - 1):
                return self._create_feedback(
                    agent_cmd, status="ARRIVED", obs=obs, reason="Final destination reached successfully."
                )
            else:
                return self._create_feedback(
                    agent_cmd, status="WAYPOINT_REACHED", obs=obs, reason="Intermediate waypoint reached."
                )
                
        # 3. Otherwise, we are just driving normally toward the target.
        return self._create_feedback(
            agent_cmd, 
            status="EXECUTED", 
            obs=obs,
            reason="Moving toward waypoint safely."
        )

    def _create_feedback(self, cmd: AgentNavigationCommand, status: str, obs: SimulatorObservation, 
                         replan: bool = False, human_help: bool = False, reason: str = "") -> NavigationFeedback:
        """Helper to build the feedback report quickly."""
        
        # Check if we saw an obstacle so we can report its distance
        obstacle_detected = False
        obs_dist = None
        for obstacle in obs.obstacles:
            if obstacle.get("blocking", False) or obstacle.get("is_blocking", False):
                obstacle_detected = True
                obs_dist = obstacle.get("distance_m")
                break
                
        return NavigationFeedback(
            run_id=cmd.run_id,
            command_id=cmd.command_id,
            status=status,
            current_waypoint=obs.current_waypoint or "unknown",
            target_waypoint=cmd.waypoint_id,
            distance_to_target_m=obs.distance_to_target_m,
            obstacle_detected=obstacle_detected,
            obstacle_distance_m=obs_dist,
            replan_required=replan,
            human_assistance_required=human_help,
            reason=reason,
            recommended_next_action="REPLAN" if replan else None
        )