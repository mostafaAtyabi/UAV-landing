
import torch
from time import sleep
import cv2
import numpy as np
from controller import Robot, Motor, Camera, Compass, GPS, Gyro, InertialUnit, Keyboard, LED
import math
import threading


length = 640
width = 400
thread_running = False

robot = Robot()
timestep = int(robot.getBasicTimeStep())
camera = robot.getDevice("camera")
camera.enable(timestep)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model.to(device)  
model.eval()

gps = robot.getDevice("gps")
gps.enable(timestep)

c_roll_disturbance = 0.0
c_pitch_disturbance = 0.0
c_camera_pitch_position = 0.0
c_yaw_disturbance = 0
c_target_altitude = 0
camera_pitch_position = 0
land = False

def sign(x):
    return (x > 0) - (x < 0)

def clamp(value, low, high):
    return max(min(value, high), low)

def create_mask(image):
    image = cv2.cvtColor(image,cv2.COLOR_BGR2HLS)
    lower = np.uint8([0, 200, 0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(image, lower, upper)

    lower = np.uint8([10, 0,   100])
    upper = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(image, lower, upper)
    mask = cv2.bitwise_or(white_mask, yellow_mask)

    # cv2.imshow("mask", mask)
    # cv2.waitKey(1)  
    return mask

def find_H(image):
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            cX, cY = 0, 0

        output_image = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)  
        cv2.circle(output_image, (cX, cY), 10, (0, 0, 255), -1)
        cv2.putText(output_image, "Center", (cX - 20, cY - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.imshow("find_H", output_image)
        cv2.waitKey(1) 
        return(cX, cY) 

def find_circle(gray):
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,  # Inverse ratio of the accumulator resolution to the image resolution
        minDist=500,  # Minimum distance between detected centers
        param1=70,  # Higher threshold for Canny edge detector
        param2=50,  # Accumulator threshold for circle detection
        minRadius=10,  # Minimum radius to be detected
        maxRadius=0  # Maximum radius to be detected (0 means no max limit)
    )
    output_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
   
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output_image, (x, y), r, (0, 255, 0), 4)
            cv2.circle(output_image, (x, y), 10, (0, 0, 255), -1)
            cv2.putText(output_image, "Center", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
      
    cv2.imshow("find_circle", output_image)
    cv2.waitKey(1)

def second_step():
    global c_roll_disturbance, c_pitch_disturbance, c_camera_pitch_position, c_yaw_disturbance, c_target_altitude, land

    while True:
        sleep(1)
        image = camera.getImageArray()
        if image:
            image_np = np.array(image, dtype=np.uint8)
            img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
            img_rgb = cv2.resize(img_rgb, (length, width))
            img_rgb = cv2.flip(img_rgb, 1)
            
            mask = create_mask(img_rgb)

            center_x , center_y = find_H(mask)
            # find_circle(mask)

            position = gps.getValues() 
            height = position[2]
            print(height)
            
            c_target_altitude = - 0.3
            c_roll_disturbance = clamp(-(center_x - length/2) * 0.05, -0.8,0.8)
            c_pitch_disturbance = clamp((center_y - width/2) * 0.02 , -1.4, 1.4)

            if center_x < length * 0.35 or center_x > length * 0.65:
                c_yaw_disturbance = clamp(-(center_x - length/2) * 0.001, -0.3, 0.3)

            if height < 1.5:
                c_roll_disturbance = 0
                c_pitch_disturbance = 0  
                land = True
                return


def image_processing():
    global c_roll_disturbance, c_pitch_disturbance, c_camera_pitch_position, c_yaw_disturbance, c_target_altitude, land
    first_step = True
    c_camera_pitch_position = 0.8  + camera_pitch_position
    
    while True:
        sleep(1)
        image = camera.getImageArray()
        if image:
            image_np = np.array(image, dtype=np.uint8)
            img_rgb = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)
            img_rgb = cv2.resize(img_rgb, (length, width))
            img_rgb = cv2.flip(img_rgb, 1) 
            results = model(img_rgb)            
   
            highest_confidence_result = None
            max_confidence = 0
            for result in results.pandas().xyxy[0].to_dict(orient='records'):
                confidence = result['confidence']
                if confidence > max_confidence:
                    max_confidence = confidence
                    highest_confidence_result = result
            
            if highest_confidence_result:
                bbox = highest_confidence_result['xmin'], highest_confidence_result['ymin'], highest_confidence_result['xmax'], highest_confidence_result['ymax']
                label = highest_confidence_result['name']
                confidence = highest_confidence_result['confidence']

                center_x = int((bbox[0] + bbox[2]) / 2)
                center_y = int((bbox[1] + bbox[3]) / 2)
                cv2.circle(img_rgb, (center_x, center_y), radius=5, color=(0, 0, 255), thickness=-1)

                cv2.putText(img_rgb, f'{label} {confidence:.2f}', (int(bbox[0]), int(bbox[1] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                if first_step:
                    c_yaw_disturbance = clamp(-(center_x - length/2) * 0.01, -0.2, 0.2)
                    
                    if center_y < 0.65 * width:  
                        c_pitch_disturbance = -2
                    else:   
                        c_pitch_disturbance = 2.1

                    if center_y > 0.6 * width and c_camera_pitch_position + camera_pitch_position  < 1.36:  
                        c_camera_pitch_position += 0.15  
                        print("camera_pitch_position change", c_camera_pitch_position) 

                    if c_camera_pitch_position + camera_pitch_position >= 1.36:
                        first_step = False
                else:
                    position = gps.getValues() 
                    height = position[2]
                    print(height)
                    if height <= 10:
                        second_step()
                        return
                    else:
                        c_target_altitude = - 1

                    c_yaw_disturbance = clamp(-(center_x - length/2) * 0.01, -0.2, 0.2)
                    c_pitch_disturbance = clamp((center_y - width/2) * 0.02, -1.5, 1.5)
            else:
                c_yaw_disturbance = -0.3
                    
            
            cv2.imshow("Black and White Image", img_rgb)
            cv2.waitKey(1)




def main():  
    global thread_running ,camera_pitch_position
    imu = robot.getDevice("inertial unit")
    imu.enable(timestep)

    gyro = robot.getDevice("gyro")
    gyro.enable(timestep)

    camera_pitch_motor = robot.getDevice("camera pitch")

    front_left_motor = robot.getDevice("front left propeller")
    front_right_motor = robot.getDevice("front right propeller")
    rear_left_motor = robot.getDevice("rear left propeller")
    rear_right_motor = robot.getDevice("rear right propeller")
    
    motors = [front_left_motor, front_right_motor, rear_left_motor, rear_right_motor]
    for motor in motors:
        motor.setPosition(float('inf'))
        motor.setVelocity(1.0)

    keyboard = Keyboard()
    keyboard.enable(timestep)

    while robot.step(timestep) != -1:
        if robot.getTime() > 1.0:
            break

    k_vertical_thrust = 68.5
    k_vertical_p = 3.0
    k_roll_p = 50.0
    k_pitch_p = 30.0
    target_altitude = 1.0
    camera_pitch_position = 0.0 

    thread = threading.Thread(target = image_processing)
    
    # Main control loop
    while robot.step(timestep) != -1:
        roll = imu.getRollPitchYaw()[0]
        pitch = imu.getRollPitchYaw()[1]
        roll_acceleration = gyro.getValues()[0]
        pitch_acceleration = gyro.getValues()[1]
        camera_pitch_motor.setPosition(-0.1 * pitch_acceleration)

        roll_disturbance = 0.0
        pitch_disturbance = 0.0
        yaw_disturbance = 0.0
        target_altitude = 0.59
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
            target_altitude += 0.5
        elif key == (Keyboard.SHIFT + Keyboard.DOWN):
            target_altitude -= 0.5
        elif key == ord('T') and not thread_running:
            thread.start()
            thread_running = True
        elif key == ord('C'): 
            thread.join()
            return 0

        global c_roll_disturbance ,c_pitch_disturbance ,c_camera_pitch_position, c_yaw_disturbance, c_target_altitude, land
        camera_pitch_motor.setPosition(clamp(camera_pitch_position + c_camera_pitch_position,0,1.5))
        roll_input = k_roll_p * clamp(roll, -1.0, 1.0) + roll_acceleration + roll_disturbance + c_roll_disturbance
        pitch_input = k_pitch_p * clamp(pitch, -1.0, 1.0) + pitch_acceleration + pitch_disturbance + c_pitch_disturbance
        yaw_input = yaw_disturbance + c_yaw_disturbance
        vertical_input = k_vertical_p * math.pow(target_altitude + c_target_altitude, 3.0)

        if land:
            front_left_motor_input = 0
            front_right_motor_input = 0
            rear_left_motor_input = 0
            rear_right_motor_input = 0
        else:
            front_left_motor_input = k_vertical_thrust + vertical_input - roll_input + pitch_input - yaw_input
            front_right_motor_input = k_vertical_thrust + vertical_input + roll_input + pitch_input + yaw_input
            rear_left_motor_input = k_vertical_thrust + vertical_input - roll_input - pitch_input + yaw_input
            rear_right_motor_input = k_vertical_thrust + vertical_input + roll_input - pitch_input - yaw_input
        
        front_left_motor.setVelocity(front_left_motor_input)
        front_right_motor.setVelocity(-front_right_motor_input)
        rear_left_motor.setVelocity(-rear_left_motor_input)
        rear_right_motor.setVelocity(rear_right_motor_input)


        



if __name__ == "__main__":
    main()
    
