from ultralytics import YOLO
import cv2
import numpy as np

class TrashCanModel:
    def __init__(self):
        # Load the trained model
        self.model = YOLO('YOLO.pt')

    def predict(self, image):
        # Resize image to 640x640
        image_resized = cv2.resize(image, (640, 640))
        
        # Run inference
        results = self.model(image_resized)
        
        # Process results
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = box.cls[0]
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class': int(cls),
                    'class_name': self.model.names[int(cls)]
                })
        
        # Draw bounding boxes on the image
        for det in detections:
            cv2.rectangle(image_resized, (det['box'][0], det['box'][1]), (det['box'][2], det['box'][3]), (0, 255, 0), 2)
            cv2.putText(image_resized, f"{det['class_name']} {det['confidence']:.2f}", 
                        (det['box'][0], det['box'][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Convert the image back to RGB (from BGR)
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        
        return detections, image_rgb