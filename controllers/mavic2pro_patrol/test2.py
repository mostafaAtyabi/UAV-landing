
import torch
import time
import cv2
import numpy as np
import os

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the YOLOv5 model to the specified device
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
model.to(device)  # Move the model to the GPU if available
model.eval()

start_time = time.time()

# Ensure the output directory exists
output_dir = 'E:\\UNI\\8\\project\\images\\capture\\'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process images
for i in range(1, 13):
    img_path = f'E:\\UNI\\8\\project\\images\\capture\\{i}.jpg'  # Path to your image
    
    # Load the image using OpenCV
    img = cv2.imread(img_path)  # img is now a NumPy array
    
    if img is None:
        print(f"Warning: Image {img_path} could not be loaded.")
        continue  # Skip to the next image if the current one could not be loaded
    
    # Convert image from BGR (OpenCV default) to RGB (expected by YOLOv5)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Perform inference
    results = model(img_rgb)  # The model handles device internally

    # Draw bounding boxes and labels on the image
    for result in results.pandas().xyxy[0].to_dict(orient='records'):
        # Extract bounding boxes and draw on image
        bbox = result['xmin'], result['ymin'], result['xmax'], result['ymax']
        label = result['name']
        confidence = result['confidence']
        
        # Draw bounding box
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
        cv2.putText(img, f'{label} {confidence:.2f}', (int(bbox[0]), int(bbox[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Save the resulting image with bounding boxes
    output_path = f'E:\\UNI\\8\\project\\images\\capture\\{i}_result.jpg'
    
    if not cv2.imwrite(output_path, img):
        print(f"Error: Image {output_path} could not be saved.")
    else:
        print(f"Saved: {output_path}")

end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
















# import torch
# import time
# from PIL import Image

# # Check if CUDA is available and set the device accordingly
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # Load the YOLOv5 model to the specified device
# model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
# model.to(device)  # Move the model to the GPU if available
# model.eval()

# start_time = time.time()

# # Process images
# for i in range(1, 13):
#     img_path = f'E:\\UNI\\8\\project\\images\\capture\\{i}.jpg'  # Path to your image

#     # Load the image
#     img = Image.open(img_path)
    
#     # Perform inference
#     results = model(img)  # The model handles device internally

#     # Save the resulting image with bounding boxes
#     results.save()  # Save the image with bounding boxes

# end_time = time.time()

# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time:.2f} seconds")




# import torch
# import time

# model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt') 

# model.eval()



# start_time = time.time()
# for i in range(1, 13):
#     img = f'E:\\UNI\\8\\project\\images\\capture\\{i}.jpg'  # replace with the path to your image
#     results = model(img)
#     # results.print()  # print results to the console
#     results.save()   # save the resulting image with bounding boxes
#     # results.show()   # display the image


# end_time = time.time()

# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time:.2f} seconds")