import cv2
import pandas as pd
from datetime import datetime
from ultralytics import YOLO
import time

# ==========================================
# CONFIGURATION
# ==========================================
# Replace '0' with a video file path (e.g., "traffic.mp4") if needed
VIDEO_SOURCE = 0 
CONFIDENCE_THRESHOLD = 0.35
ALERT_THRESHOLD = 20  # Alert if more than 20 vehicles detected

# Load the VisDrone Trained Model
print("Loading AI Model... please wait.")
model = YOLO('model.pt') 

# Initialize Video Capture
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Check if video opened successfully
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

# Font settings for display
font = cv2.FONT_HERSHEY_SIMPLEX
log_data = []

print("Starting Traffic AI. Press 'q' to quit.")

while True:
    start_time = time.time()
    
    # 1. Read Frame
    success, frame = cap.read()
    if not success:
        print("Video ended or failed to read.")
        break

    # 2. Run Detection
    results = model(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)
    
    # 3. Process Detections
    # UPDATED: Use counters that match our mapped groups
    counts = {"car": 0, "truck": 0, "bus": 0, "two_wheeler": 0, "pedestrian": 0}
    total_vehicles = 0
    
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls_id]
        
        # --- CLASS MAPPING LOGIC ---
        # Map VisDrone specific classes to our report categories
        report_name = class_name
        if class_name in ['people', 'pedestrian']:
            report_name = "pedestrian"
        elif class_name in ['bicycle', 'motor', 'tricycle', 'awning-tricycle']:
            report_name = "two_wheeler"
        
        # Filter: Only count if it matches our keys
        if report_name in counts:
            counts[report_name] += 1
            
            # Don't count pedestrians as "Traffic Vehicles" for congestion
            if report_name != "pedestrian":
                total_vehicles += 1
            
            # Draw Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Color coding: Red for people, Green for vehicles
            color = (0, 0, 255) if report_name == 'pedestrian' else (0, 255, 0)
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10), font, 0.5, color, 2)

    # 4. Add On-Screen Dashboard
    # Background for text
    cv2.rectangle(frame, (0, 0), (320, 150), (0, 0, 0), -1)
    
    # Display counts
    y_pos = 30
    cv2.putText(frame, f"Total Vehicles: {total_vehicles}", (10, y_pos), font, 0.7, (255, 255, 255), 2)
    y_pos += 30
    cv2.putText(frame, f"Cars: {counts['car']} | Trucks: {counts['truck']}", (10, y_pos), font, 0.5, (200, 200, 200), 1)
    y_pos += 25
    cv2.putText(frame, f"Buses: {counts['bus']} | 2-Wheel: {counts['two_wheeler']}", (10, y_pos), font, 0.5, (200, 200, 200), 1)
    y_pos += 25
    cv2.putText(frame, f"Pedestrians: {counts['pedestrian']}", (10, y_pos), font, 0.5, (200, 200, 200), 1)
    
    # Alert Logic
    if total_vehicles > ALERT_THRESHOLD:
        cv2.putText(frame, "HIGH CONGESTION!", (10, 140), font, 0.7, (0, 0, 255), 2)
        # Log event
        log_data.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Count": total_vehicles,
            "Status": "Congested"
        })

    # 5. Display the Frame
    cv2.imshow("Apex Research - Traffic AI Monitor", frame)

    # 6. Exit Logic (Press 'q')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Save Report on Exit
if log_data:
    df = pd.DataFrame(log_data)
    df.to_csv("traffic_log.csv", index=False)
    print("Log saved to 'traffic_log.csv'")
print("System Stopped.")