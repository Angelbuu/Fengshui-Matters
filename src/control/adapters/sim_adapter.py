import time
from src.control.adapters.base_adapter import BaseAdapter
from src.navigation.schemas import RobotCommand, SimulatorObservation, Pose, Velocity

class SimAdapter(BaseAdapter):
    """
    A temporary fake delivery driver (adapter). 
    Pretend to send commands and return fake safe data!
    """
    
    def __init__(self):
        # We will just store the last command in memory so we can check it in tests
        self.last_command_received = None
        
    def send_command(self, command: RobotCommand) -> bool:
        # Step 1: Pretend we sent it over a network
        self.last_command_received = command
        print(f"[SimAdapter] Sent command to simulator: {command.action} for {command.duration_s} seconds")
        return True
        
    def get_observation(self) -> SimulatorObservation:
        # Step 2: Pretend the robot is sitting perfectly still at the start (0,0)
        fake_pose = Pose(x_m=0.0, y_m=0.0, yaw_rad=0.0)
        fake_velocity = Velocity(vx_mps=0.0, vy_mps=0.0, wz_rps=0.0)
        
        # Step 3: Box it up into the exact structured object we promised
        observation = SimulatorObservation(
            run_id="fake_run",
            command_id="fake_cmd",
            timestamp_ms=int(time.time() * 1000),
            sim_time_s=0.0,
            robot_status="RUNNING",
            pose=fake_pose,
            velocity=fake_velocity,
            heading_error_rad=0.0,
            obstacles=[] # The path is completely clear in our fake world
        )
        return observation