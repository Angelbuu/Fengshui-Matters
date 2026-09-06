import traceback

from src.llm_agent_destination import build_resolver
from src.agent import (
    handle_destination_decision,
    decide_next_action,
    request_initial_route,
    create_next_waypoint_command,
    handle_navigation_feedback,
)
from src.control.safety_supervisor import SafetySupervisor
from src.navigation.route_evaluator import RouteEvaluator
from src.control.adapters.sim_adapter import SimAdapter
from src.navigation.route_planner import LOCATIONS

def run_hospital_robot():
    print("========================================")
    print(" FENGSHUI-MATTERS: FULL SYSTEM ONLINE")
    print("========================================\n")
    
    print("[Boot] Initializing LLM Resolver (Groq)...")
    resolver = build_resolver()
    
    print("[Boot] Initializing Navigation & Simulator...")
    adapter = SimAdapter(mode="fake")
    safety_gate = SafetySupervisor()
    evaluator = RouteEvaluator()
    
    print("\n========================================")
    print("Robot is ready in the lobby.")
    user_input = input("Visitor instruction: ")
    print("========================================\n")
    
    # 1. LLM understands intent
    print(f"[Agent A] Analyzing: '{user_input}'")
    decision = resolver.resolve_destination(user_input)
    print(f"[Agent A] Output: Intent={decision.intent.name}, Destination={decision.destination}")
    
    clarification_rounds = 0
    
    while decision.intent.name != "NAVIGATE":
        if clarification_rounds >= 5:
            print("\n[Robot says]: (System) Maximum clarification rounds reached. Resetting.")
            return
            
        print(f"\n[Robot says]: {decision.visitor_message}")
        
        if decision.intent.name == "UNKNOWN":
            print("Ending conversation.")
            return
            
        user_input = input("\nVisitor response: ")
        print(f"\n[Agent A] Analyzing: '{user_input}'")
        decision = resolver.resolve_destination(user_input)
        print(f"[Agent A] Output: Intent={decision.intent.name}, Destination={decision.destination}")
        
        clarification_rounds += 1
        
    # 2. Start Agent B state machine
    state = handle_destination_decision(decision)
    
    # 3. The Master Loop
    while True:
        action = decide_next_action(state)
        
        if action == "NAVIGATE_TO_DESTINATION":
            print(f"\n[Agent B] Requesting route to {state.destination}...")
            state = request_initial_route(state, "lobby")
            print(f"[Agent B] Route planned: {state.current_route}")
            
        elif action == "SEND_NEXT_WAYPOINT":
            is_valid, agent_cmd = create_next_waypoint_command(state)
            if not is_valid:
                print(f"[Validator] ERROR: {agent_cmd}")
                break
                
            print(f"\n[Navigation] Targeting waypoint: {agent_cmd.waypoint_id}")
            
            # Fetch observation and check safety
            obs = adapter.get_observation()
            robot_cmd = safety_gate.evaluate_command(agent_cmd, obs)
            
            # Send safe command to motors
            adapter.send_command(robot_cmd)
            
            # --- MOCK PHYSICS FOR MVP HACKATHON ---
            # Because our prototype fake physics doesn't steer, we just teleport 
            # the fake robot to the waypoint to prove the software pipeline works.
            target_wp = LOCATIONS[agent_cmd.waypoint_id]
            adapter._fake_x_m = target_wp.x_m
            adapter._fake_y_m = target_wp.y_m
            
            # Check progress
            updated_obs = adapter.get_observation()
            updated_obs.distance_to_target_m = 0.1 # Successfully arrived!
            updated_obs.current_waypoint = agent_cmd.waypoint_id
            
            was_stopped = (robot_cmd.action == "STOP")
            feedback = evaluator.evaluate_progress(agent_cmd, updated_obs, was_stopped_by_safety=was_stopped)
            
            print(f"[Evaluator] Status: {feedback.status}")
            state = handle_navigation_feedback(state, feedback)
            
        elif action == "ARRIVED":
            print("\n========================================")
            print("[Robot says]: We have arrived at your destination! Have a great day.")
            print("========================================")
            break
            
        else:
            print(f"\n[Agent B] Halting due to status: {action}")
            break

if __name__ == "__main__":
    try:
        run_hospital_robot()
    except Exception as e:
        traceback.print_exc()