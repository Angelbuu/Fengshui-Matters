import traceback
from agent_graph import build_agent_graph
from control.adapters.sim_adapter import SimAdapter

def run_hospital_robot():
    print("========================================")
    print(" FENGSHUI-MATTERS: FULL SYSTEM ONLINE")
    print("========================================\n")
    
    graph = build_agent_graph()
    adapter = SimAdapter(mode="fake")
    
    print("Robot is ready in the lobby.")
    user_input = input("Visitor instruction: ")
    
    try:
        clarification_rounds = 0
        while True:
            # LangGraph takes over the execution
            result = graph.invoke({
                "user_text": user_input,
                "adapter": adapter,
            })
            
            print("\n========================================")
            print(f"[Robot says]: {result['final_message']}")
            print("========================================")
            
            agent_status = result["agent_state"].status
            if agent_status == "WAITING_FOR_CLARIFICATION":
                clarification_rounds += 1
                if clarification_rounds >= 5:
                    print("\n[Robot says]: (System) Maximum clarification rounds reached. Resetting.")
                    break
                user_input = input("\nVisitor response: ")
            else:
                break
                
    finally:
        adapter.close()

if __name__ == "__main__":
    try:
        run_hospital_robot()
    except Exception as e:
        traceback.print_exc()