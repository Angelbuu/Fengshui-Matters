from abc import ABC, abstractmethod
from src.navigation.schemas import RobotCommand, SimulatorObservation

class BaseAdapter(ABC):
    """
    This is a 'blueprint' (Abstract Base Class). 
    It forces any future simulator, real robot, or fake test connection 
    to use these exact two methods.
    """
    
    @abstractmethod
    def send_command(self, command: RobotCommand) -> bool:
        """Sends the safe movement command to the hardware/simulator."""
        pass
        
    @abstractmethod
    def get_observation(self) -> SimulatorObservation:
        """Gets the latest camera and pose data from the hardware/simulator."""
        pass