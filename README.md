# Fengshui Matters Agent

A vision-language-action agent built for the 2026 Agentic AI Hackathon. 

Our solution leverages Agentic AI to power an autonomous hospital navigation system. By parsing raw human language into semantic intents, our agents plan optimal routes and autonomously drive a simulated robot to its destination, adapting to physical constraints safely.

## Team Members
* Angel Bu Tong Mei
* Gan Wei Siang
* Keak Jie Yi
* Teoh Ke Yi

## Tech Stack
* **Language**: Python 3
* **AI/Orchestration**: LangGraph & LangChain
* **LLMs**: AWS Bedrock & Groq

---

## Environment Setup

We strongly recommend using a virtual environment to manage dependencies.

**1. Create and activate a virtual environment:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Configure secrets:**
Copy the example environment file and fill in your API keys (e.g., Groq API key, AWS profiles):
```bash
cp .env.example .env
```

**4. Path Variables:**
When running the scripts, you must set the `PYTHONPATH` to the `src` directory so the internal module imports resolve correctly:
```bash
export PYTHONPATH=src
```

## How to Run

To run the complete end-to-end integration loop (Agentic AI + Navigation + Control):

```bash
# Ensure you are at the repository root
PYTHONPATH=src python3 src/main.py
```

1. Wait for the system to boot and initialize the LLM Resolver and Simulator.
2. When prompted, enter a semantic visitor instruction (e.g., `"Take me to x-ray"`).
3. The AI Agent will parse the intent, plan a route, and begin dispatching physical `MOVE` commands to the simulated robot until it reaches the final destination!

To run the test suite:
```bash
PYTHONPATH=src python3 -m pytest tests/
```

## Codebase Overview

Our codebase is modular, cleanly separating the AI reasoning (Agent), the spatial logic (Navigation), and the physical constraints (Control).

- `src/main.py`: The master loop that integrates all sub-systems, capturing user input and orchestrating the agent state machine.
- `src/llm_agent_destination.py`: Houses the core Agentic AI (Agent A) that interfaces with the LLM to translate raw human text into actionable semantic destinations.
- `src/agent.py` & `src/agent_state.py`: The state machine (Agent B) responsible for the "acting" and "adapting" phases. It decides when to plan a route, request commands, or yield to safety.
- `src/navigation/route_planner.py`: Contains the semantic map of the hospital (e.g., corridors, radiology) and calculates the optimal waypoint route.
- `src/navigation/command_validator.py`: A rigid safety boundary that validates JSON outputs from the Agent into strictly typed Pydantic models.
- `src/navigation/route_evaluator.py`: Evaluates the progress of the robot after each movement (e.g., flagging when a waypoint is reached or a route is blocked).
- `src/control/safety_supervisor.py`: The deterministic safety gate that intercepts intended commands and restricts physical boundaries (like maximum velocity).
- `src/control/adapters/`: Contains the `sim_adapter.py` and `orca_simulation.py` scripts that run the underlying physics engine, providing realistic feedback and latency constraints.

## Agentic AI Architecture

In alignment with the hackathon's core criteria, our Agentic AI demonstrates the following behaviors:

1. **Planning:** The AI translates vague requests (like "I feel sick") into a concrete goal, maps the hospital layout, and formulates a multi-step waypoint route.
2. **Acting:** The agent decomposes the high-level plan into explicit, physical skill calls (`MOVE`), bridging the digital-to-physical gap.
3. **Adapting:** By running a continuous closed-loop evaluation against the `SimulatorObservation`, the system reacts to blocked paths and safety interventions dynamically, demonstrating robust error recovery.
4. **Multilingual Inclusivity & Safety Bounding:** The LLM natively processes non-English instructions (e.g., Mandarin) and automatically replies in the user's spoken language. Furthermore, the Agent enforces a strict 5-round limit on clarification loops to prevent infinite state-machine trapping, ensuring a highly robust and user-friendly experience.
