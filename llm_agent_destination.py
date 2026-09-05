import os

from enum import Enum
from pathlib import Path
from typing import Literal, Protocol

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

class Intent(str, Enum):
    NAVIGATE = "NAVIGATE"
    CLARIFY = "CLARIFY"
    UNKNOWN = "UNKNOWN"

DestinationID = Literal[
    "radiology",
    "pharmacy",
    "consultation_room",
    "elevator",
    "restroom",
    "reception",
    "ward_5K",
]


SUPPORTED_DESTINATIONS = {
    "radiology": (
        "Radiology department for X-rays, scans, "
        "and medical imaging."
    ),

    "pharmacy": (
        "Pharmacy where visitors collect "
        "prescribed medication."
    ),

    "consultation_room": (
        "Consultation room where patients attend "
        "medical consultations."
    ),

    "elevator": (
        "Public elevator or lift."
    ),

    "restroom": (
        "Public restroom or toilet facilities."
    ),

    "reception": (
        "Hospital reception or front desk."
    ),

    "ward_5K": (
        "Ward 5K, the only hospital ward supported "
        "by this navigation prototype."
    ),
}

class DestinationDecision(BaseModel):
    """
    Validated output from Agent A -> Agent B.

    Example:

    {
        "intent": "NAVIGATE",
        "destination": "radiology",
        "confidence": 0.98,
        "needs_clarification": false,
        "candidates": [],
        "visitor_message": "I'll guide you to Radiology."
    }
    """

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
    )

    intent: Intent

    destination: DestinationID | None = None

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )

    needs_clarification: bool

    candidates: list[DestinationID] = Field(
        default_factory=list
    )

    visitor_message: str = Field(
        min_length=1,
        max_length=180,
    )

    @model_validator(mode="after")
    def validate_semantics(self):

        if self.intent is Intent.NAVIGATE:

            if self.destination is None:
                raise ValueError(
                    "NAVIGATE requires a destination."
                )

            if self.needs_clarification:
                raise ValueError(
                    "NAVIGATE cannot require clarification."
                )

            if self.candidates:
                raise ValueError(
                    "NAVIGATE cannot contain candidates."
                )

        elif self.intent is Intent.CLARIFY:

            if self.destination is not None:
                raise ValueError(
                    "CLARIFY cannot select a destination."
                )

            if not self.needs_clarification:
                raise ValueError(
                    "CLARIFY requires "
                    "needs_clarification=True."
                )

        elif self.intent is Intent.UNKNOWN:

            if self.destination is not None:
                raise ValueError(
                    "UNKNOWN cannot select a destination."
                )

            if self.needs_clarification:
                raise ValueError(
                    "UNKNOWN cannot require clarification."
                )

            if self.candidates:
                raise ValueError(
                    "UNKNOWN cannot contain candidates."
                )

        return self

SYSTEM_PROMPT = f"""
You are Agent A, the destination-understanding component
of a hospital navigation robot.

Your ONLY responsibility is to understand where the
visitor wants to go.

You must return a DestinationDecision.


SUPPORTED DESTINATIONS

{SUPPORTED_DESTINATIONS}


HOSPITAL INFORMATION

There is exactly ONE supported ward:

Ward 5K

Its canonical destination ID is:

ward_5K

Do NOT invent other wards.

Do NOT interpret "5K" as 5,000 wards.


SEMANTIC UNDERSTANDING

Understand the visitor's meaning rather than relying
only on exact keywords.

Examples:

"I need an X-ray."
-> radiology

"Where can I get my bones scanned?"
-> radiology

"I need to collect my prescribed medicine."
-> pharmacy

"Where is the lift?"
-> elevator

"I need the toilet."
-> restroom

"Take me back to the front desk."
-> reception

"Take me to Ward 5K."
-> ward_5K


NAVIGATE

Return NAVIGATE when exactly one supported destination
can be determined.

Requirements:

- destination must contain exactly one canonical
  destination ID.
- needs_clarification must be false.
- candidates must be empty.


CLARIFY

Return CLARIFY when the visitor's request is ambiguous
or does not contain enough information to safely
determine one destination.

Do not guess.

Example:

"I need to see the doctor."

This could refer to:

- consultation_room
- ward_5K

Therefore:

- intent = CLARIFY
- destination = null
- needs_clarification = true
- candidates should contain the plausible destinations
- visitor_message should ask one short clarification
  question.


UNKNOWN

Return UNKNOWN when the visitor clearly requests a
destination that is not supported.

Example:

"Take me to Starbucks."

-> UNKNOWN


IMPORTANT RULES

- Use semantic understanding.
- Do not rely only on exact keyword matching.
- Never invent destinations.
- Only use canonical destination IDs from the
  supported destination list.
- Ward 5K is the only supported ward.
- Never invent another ward.
- Never map an unsupported location to the
  closest-sounding supported destination.
- Ask for clarification rather than guessing.
- Keep visitor_message short and natural.
- confidence must be between 0 and 1.
- confidence is metadata only.
- Do not expose hidden reasoning or chain-of-thought.
""".strip()

class DestinationModel(Protocol):

    def decide(
        self,
        user_text: str,
    ) -> DestinationDecision:
        ...

class GroqDestinationModel:

    def __init__(
        self,
        api_key: str,
        model_name: str,
    ):

        if not api_key.strip():
            raise ValueError(
                "GROQ_API_KEY is empty."
            )

        chat = ChatGroq(
            api_key=api_key,
            model=model_name,
            temperature=0,
            timeout=30,
            max_retries=1,
        )

        self.structured_model = (
            chat.with_structured_output(
                DestinationDecision
            )
        )

    def decide(
        self,
        user_text: str,
    ) -> DestinationDecision:

        result = self.structured_model.invoke([
            (
                "system",
                SYSTEM_PROMPT,
            ),
            (
                "human",
                user_text,
            ),
        ])

        return result

class DestinationResolver:

    def __init__(
        self,
        destination_model: DestinationModel,
    ):

        self.destination_model = destination_model

    def resolve_destination(
        self,
        user_text: str,
    ) -> DestinationDecision:
        cleaned = " ".join(
            user_text.split()
        )

        if not cleaned:

            return DestinationDecision(
                intent=Intent.CLARIFY,
                destination=None,
                confidence=1.0,
                needs_clarification=True,
                candidates=[],
                visitor_message=(
                    "Where would you like me "
                    "to take you?"
                ),
            )

        if len(cleaned) > 500:

            raise ValueError(
                "Visitor instruction is too long."
            )

        return self.destination_model.decide(
            cleaned
        )

def build_resolver() -> DestinationResolver:
    """
    Build Agent A.

    Teammate B can call this function to obtain
    the destination resolver.
    """

    load_dotenv(
        dotenv_path=ENV_PATH,
        override=True,
    )

    api_key = os.getenv(
        "GROQ_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            f"GROQ_API_KEY was not found in {ENV_PATH}"
        )

    model_name = os.getenv(
        "GROQ_MODEL",
        DEFAULT_GROQ_MODEL,
    )

    destination_model = GroqDestinationModel(
        api_key=api_key,
        model_name=model_name,
    )

    return DestinationResolver(
        destination_model=destination_model
    )

def main():

    print()
    print("Fengshui-Matters — Agent A")
    print("---------------------------")

    try:

        resolver = build_resolver()

    except Exception as exc:

        print(
            "Initialization error:",
            type(exc).__name__,
            str(exc),
        )

        return

    print("Destination resolver ready.")
    print()

    user_input = input(
        "Visitor instruction: "
    )

    try:

        decision = resolver.resolve_destination(
            user_input
        )

        print()
        print("Agent A output:")
        print()

        print(
            decision.model_dump_json(
                indent=2
            )
        )

    except Exception as exc:

        print()
        print(
            "Resolver error:",
            type(exc).__name__,
            str(exc),
        )

if __name__ == "__main__":
    main()