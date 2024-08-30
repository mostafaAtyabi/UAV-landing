from controller import Robot, Motor
from time import sleep
# Initialize the Webots Robot instance
robot = Robot()

# Get the time step of the current world
timestep = int(robot.getBasicTimeStep())

# Motor names based on the C code
motor_names = [
    "left motor 1", "left motor 2", "left motor 3", "left motor 4",
    "right motor 1", "right motor 2", "right motor 3", "right motor 4"
]

# Initialize the motors
motors = [robot.getDevice(name) for name in motor_names]

# Set the motors' position to infinity for continuous rotation
for motor in motors:
    motor.setPosition(float('inf'))

# Function to set the motor velocities for both sides
def set_velocity(left_speed, right_speed):
    for i in range(4):
        motors[i].setVelocity(left_speed)  # Left motors
    for i in range(4, 8):
        motors[i].setVelocity(right_speed)  # Right motors

# Define the speeds
forward_speed = 1.5  # Adjust the speed as necessary
backward_speed = -1.5  # Negative for backward movement
move_duration = 20000  # Adjust this for how long to move in each direction

# Main loop
while robot.step(timestep) != -1:

    # Move forward
    for _ in range(move_duration):
        set_velocity(forward_speed, forward_speed)
        # sleep(1)
        if robot.step(timestep) == -1:
            break

    # Stop briefly before changing direction
    set_velocity(0.0, 0.0)



    # Move backward
    for _ in range(move_duration):
        set_velocity(backward_speed, backward_speed)
        # sleep(1)
        if robot.step(timestep) == -1:
            break

    # Stop briefly before changing direction
    set_velocity(0.0, 0.0)

