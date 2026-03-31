import cv2

model = cv2.FaceDetectorYN.create(
    "face_detection_yunet_2023mar.onnx",
    "",
    (320, 320)
)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    model.setInputSize((w, h))

    _, faces = model.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, w, h = map(int, face[:4])
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Webcam Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
