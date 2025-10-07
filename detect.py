import cv2
import numpy as np

# Load YOLOv3-tiny model
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load class labels
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load input image
img = cv2.imread("test.jpg")

if img is None:
    print("Error: test.jpg not found or cannot be opened")
    exit()

# Get image dimensions
height, width, channels = img.shape

# Convert image to YOLO input format
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), swapRB=True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

# Store detections
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-Max Suppression to remove duplicate boxes
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw results
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(classes), 3))

if len(indexes) > 0:
    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = confidences[i]
        color = colors[class_ids[i]]

        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f"{label} {confidence:.2f}", (x, y - 10), font, 1, color, 2)

        # Print to console
        print(f"Detected {label} with confidence {confidence:.2f}")
else:
    print("No objects detected!")

# Save output image
cv2.imwrite("output.jpg", img)
print(" Output saved as output.jpg")

# Show smaller display window (resize for viewing only)
display_img = cv2.resize(img, (900, 600))
cv2.imshow("Object Detection", display_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
