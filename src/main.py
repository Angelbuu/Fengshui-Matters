import os
from pathlib import Path
import subprocess
import sys
import traceback

from src.llm_agent_destination import build_resolver


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASTAR_RUNNER = PROJECT_ROOT / "run_orca_astar.py"


def run_real_orca_navigation(destination: str) -> bool:
    """
    Launch the tested Orca A* physical-navigation layer.

    The destination has already been resolved by Agent A.
    """

    print()
    print("=" * 60)
    print("STARTING PHYSICAL ORCA NAVIGATION")
    print(f"Destination: {destination}")
    print("=" * 60)
    print()

    command = [
        sys.executable,
        str(ASTAR_RUNNER),
        destination,
        "--max-decisions",
        "300",
    ]

    result = subprocess.run(
        command,
        cwd=str(PROJECT_ROOT),
        env=os.environ.copy(),
    )

    return result.returncode == 0


def run_hospital_robot():
    print()
    print("========================================")
    print(" FENGSHUI-MATTERS: HOSPITAL GUIDE")
    print("========================================")
    print()

    # --------------------------------------------------------
    # Initialize teammate's LLM destination resolver
    # --------------------------------------------------------

    print("[Boot] Initializing destination resolver...")

    resolver = build_resolver()

    print("[Boot] LLM ready.")
    print("[Boot] Orca navigation will start after a destination is confirmed.")

    print()
    print("Robot is ready.")
    print()

    # --------------------------------------------------------
    # Initial visitor request
    # --------------------------------------------------------

    user_input = input(
        "Visitor instruction: "
    ).strip()

    if not user_input:
        print("No instruction received.")
        return

    print()
    print(
        f"[Agent A] Analyzing: "
        f"'{user_input}'"
    )

    decision = resolver.resolve_destination(
        user_input
    )

    print(
        f"[Agent A] Intent="
        f"{decision.intent.name}, "
        f"Destination="
        f"{decision.destination}"
    )

    # --------------------------------------------------------
    # Clarification loop from teammate's implementation
    # --------------------------------------------------------

    clarification_rounds = 0

    while decision.intent.name != "NAVIGATE":

        if clarification_rounds >= 5:
            print()
            print(
                "[Robot]: I still don't have enough "
                "information to determine where you need to go."
            )
            return

        print()
        print(
            f"[Robot]: "
            f"{decision.visitor_message}"
        )

        if decision.intent.name == "UNKNOWN":
            return

        user_input = input(
            "Visitor response: "
        ).strip()

        if not user_input:
            print(
                "[Robot]: I didn't receive a response."
            )
            return

        print()
        print(
            f"[Agent A] Analyzing: "
            f"'{user_input}'"
        )

        decision = resolver.resolve_destination(
            user_input
        )

        print(
            f"[Agent A] Intent="
            f"{decision.intent.name}, "
            f"Destination="
            f"{decision.destination}"
        )

        clarification_rounds += 1

    # --------------------------------------------------------
    # Destination confirmed
    # --------------------------------------------------------

    destination = decision.destination

    if destination is None:
        print(
            "[Robot]: I couldn't determine "
            "a destination."
        )
        return

    print()
    print(
        f"[Robot]: "
        f"{decision.visitor_message}"
    )

    print()
    print(
        f"[System] Confirmed destination: "
        f"{destination}"
    )

    # --------------------------------------------------------
    # Hand destination to YOUR tested Orca A* layer
    # --------------------------------------------------------

    success = run_real_orca_navigation(
        destination
    )

    print()

    if success:
        print("=" * 60)
        print(
            f"[Robot]: We have arrived at "
            f"{destination.replace('_', ' ')}."
        )
        print("=" * 60)

    else:
        print("=" * 60)
        print(
            "[Robot]: I couldn't complete the route safely."
        )
        print("=" * 60)


if __name__ == "__main__":
    try:
        run_hospital_robot()

    except KeyboardInterrupt:
        print()
        print("[System] Navigation cancelled.")

    except Exception:
        print()
        print("[System] Unexpected error:")
        traceback.print_exc()
