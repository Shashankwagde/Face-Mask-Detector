import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("mask_detector_mobilenetv2.keras")

face_cascade = cv2.CascadeClassifier(
    "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    raise IOError("Haar cascade XML file not loaded. Check file path.")

IMG_SIZE = 224
FONT = cv2.FONT_HERSHEY_SIMPLEX

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

print("Webcam started successfully. Press 'q' to quit.")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(60, 60)
    )

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        face = cv2.resize(face, (IMG_SIZE, IMG_SIZE))
        face = face.astype("float32") / 255.0
        face = np.expand_dims(face, axis=0)

        prediction = model.predict(face, verbose=0)
        class_id = np.argmax(prediction)

        if class_id == 0:
            label = "Mask Found"
            color = (0, 255, 0)
        else:
            label = "No Mask, Wear Mask NOW"
            color = (0, 0, 255)

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            label,
            (x, y - 10),
            FONT,
            0.8,
            color,
            2,
            cv2.LINE_AA
        )

    # Show output
    cv2.imshow("Face Mask Detector", frame)

    # Exit on 'q' or closely window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    if cv2.getWindowProperty("Face Mask Detector", cv2.WND_PROP_VISIBLE) < 1:
        break


cap.release()
cv2.destroyAllWindows()
print("Webcam closed.")
