import sys
import cv2
import numpy as np
import torch
import os




def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')
    model.to(device)  
    model.eval()


    img_bytes = sys.stdin.buffer.read()
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image_np = cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE)

    # Perform inference
    results = model(image_np)
    
    for result in results.pandas().xyxy[0].to_dict(orient='records'):
            # Extract bounding boxes and draw on image
            bbox = result['xmin'], result['ymin'], result['xmax'], result['ymax']
            label = result['name']
            confidence = result['confidence']
            
            # Draw bounding box
            cv2.rectangle(image_np, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(image_np, f'{label} {confidence:.2f}', (int(bbox[0]), int(bbox[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        

    cv2.imshow("Black and White Image", image_np)
    cv2.waitKey(1)  # Wait for 1 ms to refresh the display

if __name__ == '__main__':
    main()
