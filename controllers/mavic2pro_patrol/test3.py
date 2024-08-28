
import torch
import time
import cv2
import numpy as np
from controller import Robot, Motor, Camera, Compass, GPS, Gyro, InertialUnit, Keyboard, LED
import math
import subprocess
import os

def sign(x):
    return (x > 0) - (x < 0)

def clamp(value, low, high):
    return max(min(value, high), low)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    image_cnt = 0

    robot = Robot()
    timestep = int(robot.getBasicTimeStep())

    # Get and enable devices.
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    front_left_led = robot.getDevice("front left led")
    front_right_led = robot.getDevice("front right led")
    imu = robot.getDevice("inertial unit")
    imu.enable(timestep)
    gps = robot.getDevice("gps")
    gps.enable(timestep)
    compass = robot.getDevice("compass")
    compass.enable(timestep)
    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)
    keyboard = Keyboard()
    keyboard.enable(timestep)
    camera_roll_motor = robot.getDevice("camera roll")
    camera_pitch_motor = robot.getDevice("camera pitch")
    camera_yaw_motor = robot.getDevice("camera yaw")  # Added for yaw control

    front_left_motor = robot.getDevice("front left propeller")
    front_right_motor = robot.getDevice("front right propeller")
    rear_left_motor = robot.getDevice("rear left propeller")
    rear_right_motor = robot.getDevice("rear right propeller")
    motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]
    for motor in motors:
        motor.setPosition(float('inf'))
        motor.setVelocity(1.0)

    print("Start the drone...")

    while robot.step(timestep) != -1:
        if robot.getTime() > 1.0:
            break

    print("You can control the drone with your computer keyboard:")
    print("- 'up': move forward.")
    print("- 'down': move backward.")
    print("- 'right': turn right.")
    print("- 'left': turn left.")
    print("- 'shift + up': increase the target altitude.")
    print("- 'shift + down': decrease the target altitude.")
    print("- 'shift + right': strafe right.")
    print("- 'shift + left': strafe left.")
    print("- 'W': tilt camera up.")
    print("- 'S': tilt camera down.")
    print("- 'A': pan camera left.")
    print("- 'D': pan camera right.")

    k_vertical_thrust = 68.5
    k_vertical_offset = 0.6
    k_vertical_p = 3.0
    k_roll_p = 50.0
    k_pitch_p = 30.0

    target_altitude = 1.0

    # Variables for camera control
    camera_pitch_position = 0.0
    camera_yaw_position = 0.0
    camera_rotation_speed = 0.05  # Adjust this value to change camera rotation speed

    # Main control loop
    while robot.step(timestep) != -1:
        time = robot.getTime()
        roll = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        altitude = gps.getValues()[2]
        roll_acceleration = gyro.getValues()[0]
        pitch_acceleration = gyro.getValues()[1]

        led_state = int(time) % 2
        front_left_led.set(led_state)
        front_right_led.set(1 - led_state)

        camera_roll_motor.setPosition(-0.115 * roll_acceleration)
        camera_pitch_motor.setPosition(-0.1 * pitch_acceleration)

        roll_disturbance = 0.0
        pitch_disturbance = 0.0
        yaw_disturbance = 0.0
        key = keyboard.getKey()
        
        if key == Keyboard.UP:
            pitch_disturbance = -2.0
        elif key == Keyboard.DOWN:
            pitch_disturbance = 2.0
        elif key == Keyboard.RIGHT:
            yaw_disturbance = -1.3
        elif key == Keyboard.LEFT:
            yaw_disturbance = 1.3
        elif key == (Keyboard.SHIFT + Keyboard.RIGHT):
            roll_disturbance = -1.0
        elif key == (Keyboard.SHIFT + Keyboard.LEFT):
            roll_disturbance = 1.0
        elif key == (Keyboard.SHIFT + Keyboard.UP):
            target_altitude += 0.05
            print(f"target altitude: {target_altitude:.2f} [m]")
        elif key == (Keyboard.SHIFT + Keyboard.DOWN):
            target_altitude -= 0.05
            print(f"target altitude: {target_altitude:.2f} [m]")
        elif key == ord('W'):  # Tilt camera up
            camera_pitch_position += camera_rotation_speed
        elif key == ord('S'):  # Tilt camera down
            camera_pitch_position -= camera_rotation_speed
        elif key == ord('A'):  # Pan camera left
            camera_yaw_position += camera_rotation_speed
        elif key == ord('D'):  # Pan camera right
            camera_yaw_position -= camera_rotation_speed


        # elif key == ord('C'):  # capture
        #     image = camera.getImageArray()
        #     image_np = np.array(image, dtype=np.uint8)
        #     image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB) 
        #     image_np = cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE)
        #     path = 'E://UNI//8//project//images//capture'
        #     image_cnt += 1
        #     cv2.imwrite(os.path.join(path , f'{image_cnt}.jpg'), image_np)



        # Update camera pitch and yaw motors
        camera_pitch_motor.setPosition(camera_pitch_position)
        camera_yaw_motor.setPosition(camera_yaw_position)

        roll_input = k_roll_p * clamp(roll, -1.0, 1.0) + roll_acceleration + roll_disturbance
        pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + pitch_acceleration + pitch_disturbance
        yaw_input = yaw_disturbance
        clamped_difference_altitude = clamp(target_altitude - altitude + k_vertical_offset, -1.0, 1.0)
        vertical_input = k_vertical_p * math.pow(clamped_difference_altitude, 3.0)

        front_left_motor_input = k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
        front_right_motor_input = k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
        rear_left_motor_input = k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
        rear_right_motor_input = k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input
        
        front_left_motor.setVelocity(front_left_motor_input)
        front_right_motor.setVelocity(-front_right_motor_input)
        rear_left_motor.setVelocity(-rear_left_motor_input)
        rear_right_motor.setVelocity(rear_right_motor_input)


        if image_cnt < 100:
            image_cnt += 1
        else:
            image = camera.getImageArray()
            if image:
                image_cnt = 0
                image_np = np.array(image, dtype=np.uint8)


                _, img_encoded = cv2.imencode('.jpg', img)
                img_bytes = img_encoded.tobytes()

                # Call the external script with the image data
                proc = subprocess.Popen(['python', 'test4.py'], stdin=subprocess.PIPE)
                proc.communicate(input=img_bytes)







if __name__ == "__main__":
    main()
