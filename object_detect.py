import cv2
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    for result in results.boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        conf = float(result.conf[0])
        cls = int(result.cls[0])
        label = model.names[cls]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        text = f"{label} {conf:.2f}"

        (text_width, text_height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame, text, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.imshow("Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
