from typing import Literal, Optional

from pydantic import BaseModel


ObstacleType = Literal[
    "PERSON",
    "WHEELCHAIR",
    "CART",
    "DOOR",
    "WALL",
    "OTHER",
    "UNKNOWN",
]


class VisionResult(BaseModel):
    """
    Semantic interpretation of the robot's camera view.
    """

    obstacle_type: ObstacleType = "UNKNOWN"

    confidence: float = 0.0

    description: str = ""

    path_blocked: bool = False

    suggested_behavior: Optional[str] = None
    
class FakeVisionProvider:
    """
    Lightweight vision provider for development without
    requiring a real camera model.
    """

    def __init__(
        self,
        result: VisionResult | None = None,
    ):
        self.result = result or VisionResult()

    def analyze(
        self,
        camera_frame,
    ) -> VisionResult:
        return self.result
    
def choose_obstacle_behavior(
    vision: VisionResult,
) -> str:
    """
    Convert semantic perception into a high-level
    obstacle response.
    """

    if not vision.path_blocked:
        return "CONTINUE"

    if vision.obstacle_type in {
        "PERSON",
        "WHEELCHAIR",
    }:
        return "WAIT"

    if vision.obstacle_type in {
        "CART",
        "WALL",
    }:
        return "DETOUR"

    return "SAFE_STOP"

if __name__ == "__main__":

    person_result = VisionResult(
        obstacle_type="PERSON",
        confidence=0.95,
        description="Person standing ahead.",
        path_blocked=True,
    )

    cart_result = VisionResult(
        obstacle_type="CART",
        confidence=0.92,
        description="Hospital cart blocking corridor.",
        path_blocked=True,
    )

    clear_result = VisionResult(
        obstacle_type="UNKNOWN",
        confidence=0.9,
        description="Path appears clear.",
        path_blocked=False,
    )

    print(
        "Person:",
        choose_obstacle_behavior(person_result),
    )

    print(
        "Cart:",
        choose_obstacle_behavior(cart_result),
    )

    print(
        "Clear:",
        choose_obstacle_behavior(clear_result),
    )