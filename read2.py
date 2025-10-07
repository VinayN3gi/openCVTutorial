import cv2
import time
from datetime import datetime

# Load Haar cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Open webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

check_interval = 5  # seconds
last_check_time = 0

# Logs for each event
log_entries = []

# Metrics
looked_away_count = 0
multiple_faces_count = 0
total_checks = 0

# To track last face status (to avoid counting multiple times for same event)
last_status = "One face"

# Log start time
start_time = datetime.now()
log_entries.append({"time": start_time.strftime("%Y-%m-%d %H:%M:%S"), "status": "Recording started"})

while True:
    current_time = time.time()
    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    ret, frame = cap.read()
    if not ret:
        break

    # Show current time on video
    cv2.putText(frame, current_time_str, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if current_time - last_check_time >= check_interval:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        face_count = len(faces)
        total_checks += 1

        if face_count == 0:
            print("No face")
            status = "No face"
            if last_status != "No face":  # count only when status changes
                looked_away_count += 1
                log_entries.append({"time": current_time_str, "status": "No face detected"})
        elif face_count > 1:
            print("Multiple faces")
            status = "Multiple faces"
            if last_status != "Multiple faces":
                multiple_faces_count += 1
                log_entries.append({"time": current_time_str, "status": f"Multiple faces detected ({face_count})"})
        else:
            print('One face')
            status = "One face"

        last_status = status
        last_check_time = current_time

    # Draw rectangles around detected faces
    for (x, y, w, h) in face_cascade.detectMultiScale(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    ):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow("Webcam Feed", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        end_time = datetime.now()
        log_entries.append({"time": end_time.strftime("%Y-%m-%d %H:%M:%S"), "status": "Recording ended"})
        break

cap.release()
cv2.destroyAllWindows()

# Calculate total duration
duration_seconds = (end_time - start_time).total_seconds()
duration_minutes = duration_seconds / 60

# Final report
print("\n--- Detection Log ---")
for entry in log_entries:
    print(f"{entry['time']} - {entry['status']}")

print("\n--- Test Metrics ---")
print(f"Total test duration: {duration_minutes:.2f} minutes")
print(f"Total checks: {total_checks}")
print(f"Times looked away: {looked_away_count}")
print(f"Times multiple people detected: {multiple_faces_count}")
print(f"Focus retention: {((total_checks - looked_away_count - multiple_faces_count) / total_checks) * 100:.2f}%")
