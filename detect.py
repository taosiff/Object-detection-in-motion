import cv2
from ultralytics import YOLO
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

model = YOLO("yolov8x.pt")
model.to(device)

cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (frame_width, frame_height))

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]
prev_frame = None
motion_threshold = 25
min_motion_area = 400
tracked_objects = {}  
while cap.isOpened():
    success, img = cap.read()
    if not success:
        break
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if prev_frame is None:
        prev_frame = gray
        continue
    frame_diff = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_diff, motion_threshold, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_mask = np.zeros_like(thresh)
    motion_detected = False
    for contour in contours:
        if cv2.contourArea(contour) > min_motion_area:
            motion_detected = True
            cv2.drawContours(motion_mask, [contour], 0, 255, -1)
    
    current_frame_objects = {}
    results = model(
        img,
        stream=True,
        conf=0.35,
        iou=0.45,
        agnostic_nms=True,
        max_det=100,
        device=device
    )

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            cls_name = classNames[cls]
            
            obj_center_x = (x1 + x2) // 2
            obj_center_y = (y1 + y2) // 2
            obj_id = f"{cls_name}_{obj_center_x}_{obj_center_y}"
            
            if motion_detected:
                object_roi = motion_mask[y1:y2, x1:x2]
                if object_roi.size > 0 and np.count_nonzero(object_roi) > (object_roi.size * 0.1):
                    tracked_objects[obj_id] = {
                        "class": cls_name,
                        "conf": conf,
                        "bbox": (x1, y1, x2, y2),
                        "center": (obj_center_x, obj_center_y)
                    }
            
            current_frame_objects[obj_id] = {
                "class": cls_name,
                "conf": conf,
                "bbox": (x1, y1, x2, y2),
                "center": (obj_center_x, obj_center_y)
            }
    
    obj_ids_to_remove = []
    for obj_id in tracked_objects:
        cls_name = tracked_objects[obj_id]["class"]
        old_center = tracked_objects[obj_id]["center"]   
        best_match = None
        min_distance = float('inf')
        
        for curr_id, curr_obj in current_frame_objects.items():
            if curr_obj["class"] == cls_name:
                curr_center = curr_obj["center"]
                distance = ((curr_center[0] - old_center[0]) ** 2 + 
                           (curr_center[1] - old_center[1]) ** 2) ** 0.5
                if distance < 50 and distance < min_distance:  
                    min_distance = distance
                    best_match = curr_id
        
        if best_match:
            tracked_objects[obj_id] = current_frame_objects[best_match]
        else:
            obj_ids_to_remove.append(obj_id)
    for obj_id in obj_ids_to_remove:
        del tracked_objects[obj_id]
    for obj_id, obj_info in tracked_objects.items():
        x1, y1, x2, y2 = obj_info["bbox"]
        conf = obj_info["conf"]
        cls_name = obj_info["class"]
        color = (0, int(255 * conf), int(255 * (1 - conf)))
        label = f"{cls_name} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    motion_display = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
    motion_display = cv2.addWeighted(img, 0.8, motion_display, 0.2, 0)
    prev_frame = gray
    out.write(img)
    cv2.imshow("YOLOv8x Moving Object Detection", img)
    cv2.imshow("Motion Detection", motion_display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'): 
        tracked_objects.clear()
        print("Tracked objects reset")
cap.release()
out.release()
cv2.destroyAllWindows()