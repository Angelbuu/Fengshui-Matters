import time
from src.navigation.schemas import AgentNavigationCommand, SimulatorObservation, Pose, Velocity
from src.navigation.route_evaluator import RouteEvaluator

def test_evaluator_reports_waypoint_reached():
    evaluator = RouteEvaluator()
    
    # Arrange: AI wants to go to waypoint 1
    agent_cmd = AgentNavigationCommand(
        run_id="r1", command_id="c1", destination="radiology", 
        route_id="route1", waypoint_id="wp_1", waypoint_index=0, total_waypoints=5,
        action="MOVE_TO_WAYPOINT", max_duration_s=1.0, confidence=0.9
    )
    
    # Arrange: Observation says we are only 0.2 meters away!
    obs = SimulatorObservation(
        run_id="r1", command_id="c1", timestamp_ms=int(time.time() * 1000),
        sim_time_s=1.0, robot_status="RUNNING",
        pose=Pose(x_m=0.0, y_m=0.0, yaw_rad=0.0), velocity=Velocity(vx_mps=0.0, vy_mps=0.0, wz_rps=0.0),
        heading_error_rad=0.0, distance_to_target_m=0.2, obstacles=[]
    )
    
    # Act: Did we get stopped by safety? No (False).
    feedback = evaluator.evaluate_progress(agent_cmd, obs, was_stopped_by_safety=False)
    
    # Assert: We should tell the AI that it succeeded!
    assert feedback.status == "WAYPOINT_REACHED"
    assert feedback.replan_required is False

def test_evaluator_reports_blocked():
    evaluator = RouteEvaluator()
    
    # Arrange: AI wants to go to waypoint 1
    agent_cmd = AgentNavigationCommand(
        run_id="r1", command_id="c1", destination="radiology", 
        route_id="route1", waypoint_id="wp_1", waypoint_index=0, total_waypoints=5,
        action="MOVE_TO_WAYPOINT", max_duration_s=1.0, confidence=0.9
    )
    
    # Arrange: Observation sees a blocking cart
    obs = SimulatorObservation(
        run_id="r1", command_id="c1", timestamp_ms=int(time.time() * 1000),
        sim_time_s=1.0, robot_status="RUNNING",
        pose=Pose(x_m=0.0, y_m=0.0, yaw_rad=0.0), velocity=Velocity(vx_mps=0.0, vy_mps=0.0, wz_rps=0.0),
        heading_error_rad=0.0, distance_to_target_m=5.0, 
        obstacles=[{"id": "cart", "distance_m": 0.2, "blocking": True}]
    )
    
    # Act: Safety supervisor DID stop us! (True)
    feedback = evaluator.evaluate_progress(agent_cmd, obs, was_stopped_by_safety=True)
    
    # Assert: We should tell the AI it is blocked and needs to replan.
    assert feedback.status == "BLOCKED"
    assert feedback.replan_required is True