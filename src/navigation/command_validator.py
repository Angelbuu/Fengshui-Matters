from typing import Dict, Any, Tuple
from pydantic import ValidationError
from navigation.schemas import AgentNavigationCommand

def validate_agent_command(raw_data: Dict[str, Any]) -> Tuple[bool, AgentNavigationCommand | str]:
    """
    Checks if the raw dictionary from the Agent matches our strict rules.
    
    Inputs:
        raw_data: A standard Python dictionary holding the Agent's request.
        
    Outputs:
        Returns (True, ValidatedCommandObject) if the data is perfect.
        Returns (False, ErrorMessageString) if the data broke a rule.
    """
    try:
        # Step 1: Hand the raw data to our Pydantic schema (the bouncer).
        # The '**' unwraps the dictionary so Pydantic can read every field.
        valid_command = AgentNavigationCommand(**raw_data)
        
        # Step 2: If it didn't crash, the data is safe!
        return True, valid_command
        
    except ValidationError as e:
        # Step 3: If Pydantic found a rule violation (like a wrong destination name),
        # it raises a ValidationError. We catch it here so the program doesn't die.
        error_message = f"Command rejected because of bad data:\n{str(e)}"
        return False, error_message
        
    except Exception as e:
        # Step 4: Catch any other weird computer errors just in case.
        return False, f"Unexpected error during validation: {str(e)}"