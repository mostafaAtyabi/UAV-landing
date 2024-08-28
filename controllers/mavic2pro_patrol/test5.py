import torch
import cv2
import numpy as np
from controller import Robot, Motor, Camera, GPS, InertialUnit, Keyboard
from yolov5 import YOLOv5

def clamp(value, low, high):
    return max(min(value, high), low)

def main():
    # Initialize the robot and the YOLOv5 model
    robot = Robot()
    timestep = int(robot.getBasicTimeStep())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the camera and other devices
    camera = robot.getDevice("camera")
    camera.enable(timestep)
    front_left_motor = robot.getDevice("front left propeller")
    front_right_motor = robot.getDevice("front right propeller")
    rear_left_motor = robot.getDevice("rear left propeller")
    rear_right_motor = robot.getDevice("rear right propeller")
    motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]
    for motor in motors:
        motor.setPosition(float('inf'))
        motor.setVelocity(0.0)  # Initialize with zero velocity

    # Load YOLOv5 small model
    yolo_model = YOLOv5("path/to/yolov5s.pt", device)

    # Control parameters
    k_move = 0.5
    target_distance_threshold = 1.0  # meters
    camera_pitch_speed = 0.05
    speed_increase_interval = 50
    image_cnt = 0

    # Main control loop
    while robot.step(timestep) != -1:
        # Capture image from the camera
        image = camera.getImageArray()
        if image:
            image_np = np.array(image, dtype=np.uint8)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

            # Detect helipad
            results = yolo_model.predict(image_np)
            bboxes = results.xyxy[0].cpu().numpy()  # Get bounding boxes

            if len(bboxes) > 0:
                # Assuming the first detected object is the helipad
                x_center, y_center, width, height = bboxes[0][0], bboxes[0][1], bboxes[0][2], bboxes[0][3]
                frame_height, frame_width = image_np.shape[:2]
                
                # Calculate offsets
                x_offset = (x_center - frame_width / 2) / frame_width
                y_offset = (y_center - frame_height / 2) / frame_height
                
                # Adjust speed based on offsets
                forward_speed = k_move * (1 - abs(x_offset))
                turn_speed = k_move * x_offset

                # Adjust camera pitch if helipad is moving out of view
                if abs(y_offset) > 0.1:
                    camera_pitch_position = camera_pitch_motor.getPosition()
                    camera_pitch_motor.setPosition(camera_pitch_position - camera_pitch_speed * np.sign(y_offset))
                
                # Set motor velocities
                front_left_motor.setVelocity(forward_speed - turn_speed)
                front_right_motor.setVelocity(forward_speed + turn_speed)
                rear_left_motor.setVelocity(forward_speed - turn_speed)
                rear_right_motor.setVelocity(forward_speed + turn_speed)

                # Capture and process image every 50 iterations
                if image_cnt % speed_increase_interval == 0:
                    # Increase speed and process image
                    for motor in motors:
                        motor.setVelocity(motor.getVelocity() * 1.1)
                    # Optional: Save image
                    image_cnt = 0  # Reset image counter

                image_cnt += 1
                
            else:
                # Landing mode: Use edge detection to finalize landing
                gray_image = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray_image, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if len(contours) > 0:
                    # Process contours to check if within landing zone
                    for contour in contours:
                        if cv2.contourArea(contour) > 1000:
                            front_left_motor.setVelocity(0.0)
                            front_right_motor.setVelocity(0.0)
                            rear_left_motor.setVelocity(0.0)
                            rear_right_motor.setVelocity(0.0)
                            print("Landing...")
                            break

        # Add a delay to simulate real-time processing
        robot.step(10)

if __name__ == "__main__":
    main()
