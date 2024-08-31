from controller import Robot, Motor

robot = Robot()

timestep = int(robot.getBasicTimeStep())

motor_names = [
    "left motor 1", "left motor 2", "left motor 3", "left motor 4",
    "right motor 1", "right motor 2", "right motor 3", "right motor 4"
]

motors = [robot.getDevice(name) for name in motor_names]

for motor in motors:
    motor.setPosition(float('inf'))

def set_velocity(left_speed, right_speed):
    for i in range(4):
        motors[i].setVelocity(left_speed)  # Left motors
    for i in range(4, 8):
        motors[i].setVelocity(right_speed)  # Right motors

forward_speed = 1.5  # Adjust the speed as necessary
backward_speed = -1.5  # Negative for backward movement
move_duration = 20000  # Adjust this for how long to move in each direction

# Main loop
while robot.step(timestep) != -1:
    for _ in range(move_duration):
        set_velocity(forward_speed, forward_speed)
        if robot.step(timestep) == -1:
            break
    set_velocity(0.0, 0.0)

    # Move backward
    for _ in range(move_duration):
        set_velocity(backward_speed, backward_speed)
        if robot.step(timestep) == -1:
            break
    set_velocity(0.0, 0.0)

