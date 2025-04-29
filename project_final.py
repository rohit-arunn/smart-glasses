import cv2
import numpy as np
import pyttsx3
from time import time

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load YOLO model and COCO class labels
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Precompute layer names
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Open the webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Initialize Saliency algorithm
saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

# Tracking detected objects with timestamps to reduce repeat speech
previous_objects = {}
announced_left_objects = set()

# Constants for object tracking and TTS
frame_skip = 2  # Process every nth frame to improve speed
frame_count = 0
detection_interval = 10  # Object detection frequency in seconds
last_detection_time = 0

def interactive_speech(label, position, distance="close"):
    """Generate interactive speech for objects based on position and distance."""
    location_phrases = {
        "left": "on your left",
        "right": "on your right",
        "center": "straight ahead",
        "top": "above you",
        "bottom": "below you",
        "middle": "in front of you"
    }
    return f"There is a {distance} {label} {location_phrases.get(position.split()[1], '')}."

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image.")
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    # Resize frame for faster processing
    original_height, original_width = frame.shape[:2]
    target_width, target_height = 640, 480
    frame_resized = cv2.resize(frame, (target_width, target_height))

    # Detect salient regions at reduced resolution for efficiency
    success, saliency_map_small = saliency.computeSaliency(cv2.resize(frame_resized, (160, 120)))
    saliency_map = cv2.resize((saliency_map_small * 255).astype("uint8"), (target_width, target_height))

    # Check if detection interval has passed
    current_time = time()
    if current_time - last_detection_time < detection_interval:
        continue

    # YOLO object detection
    blob = cv2.dnn.blobFromImage(frame_resized, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Parse YOLO outputs
    class_ids, confidences, boxes = [], [], []
    current_frame_objects = {}

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                center_x, center_y, w, h = (detection[0:4] * np.array([target_width, target_height, target_width, target_height])).astype('int')
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.5)

    # Track current detected objects
    if len(indexes) > 0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]
            
            # Determine object position and estimated distance
            center_x, center_y = x + w // 2, y + h // 2
            horiz_pos = "left" if center_x < target_width // 3 else "right" if center_x > 2 * target_width // 3 else "center"
            vert_pos = "top" if center_y < target_height // 3 else "bottom" if center_y > 2 * target_height // 3 else "middle"
            position = f"{vert_pos} {horiz_pos}"
            distance = "nearby" if w * h > 30000 else "a bit farther"

            object_key = (label, position)

            # Draw bounding box and label
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame_resized, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Interactive speech output every 10 seconds
            if object_key not in previous_objects or current_time - previous_objects[object_key] >= detection_interval:
                speech_text = interactive_speech(label, position, distance)
                engine.say(speech_text)
                engine.runAndWait()
                print(f"Announced: {speech_text}")
                previous_objects[object_key] = current_time

            current_frame_objects[object_key] = current_time

    # Handle objects that have left the frame
    for obj_key in list(previous_objects):
        if obj_key not in current_frame_objects:
            if obj_key not in announced_left_objects:
                label, position = obj_key
                speech_text = f"The {label} at the {position} has left."
                engine.say(speech_text)
                engine.runAndWait()
                print(f"Announced: {speech_text}")
                announced_left_objects.add(obj_key)
            del previous_objects[obj_key]

    # Update frame display
    cv2.imshow('Object Detection', cv2.resize(frame_resized, (original_width, original_height)))
    cv2.imshow('Saliency Map', cv2.resize(saliency_map, (original_width, original_height)))

    # Update last detection time
    last_detection_time = current_time

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


