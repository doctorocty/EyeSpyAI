import cv2
from fer import FER

print("starting")

print("loading FER model")
detector = FER(mtcnn=True)
print("FER model loaded!")

print("opening webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
while True:
    ret, frame = cap.read()
    if not ret:
        print("no frame captured")
        break

    results = detector.detect_emotions(frame)

    for result in results:
        (x, y, w, h) = result["box"]
        emotions = result["emotions"]
        dominant_emotion = max(emotions, key=emotions.get)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, dominant_emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Exited")
