import cv2
import time
import datetime
import mediapipe as mp
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import os

# --- SETTINGS ---
CHECK_INTERVAL = 5
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700

# --- Init ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

start_time = time.time()
last_check_time = start_time
current_state = "green"
timeline = []
metrics = {
    "looked_away_face_count": 0,
    "multiple_faces_count": 0,
    "looked_away_eyes_count": 0,
    "timestamps": [],
}
total_checks = 0

# --- Helper: Append to timeline ---
def append_timeline(color, duration):
    if duration > 0:
        timeline.append((color, duration))

# --- Eye tracking helper ---
def check_gaze_direction(landmarks):
    LEFT_IRIS = [474, 475, 476, 477]
    RIGHT_IRIS = [469, 470, 471, 472]
    LEFT_EYE = [33, 133]
    RIGHT_EYE = [362, 263]

    left_iris_x = sum([landmarks[i].x for i in LEFT_IRIS]) / len(LEFT_IRIS)
    right_iris_x = sum([landmarks[i].x for i in RIGHT_IRIS]) / len(RIGHT_IRIS)

    left_eye_left = landmarks[LEFT_EYE[0]].x
    left_eye_right = landmarks[LEFT_EYE[1]].x
    right_eye_left = landmarks[RIGHT_EYE[0]].x
    right_eye_right = landmarks[RIGHT_EYE[1]].x

    left_ratio = (left_iris_x - left_eye_left) / (left_eye_right - left_eye_left)
    right_ratio = (right_iris_x - right_eye_left) / (right_eye_right - right_eye_left)

    return 0.35 < left_ratio < 0.65 and 0.35 < right_ratio < 0.65

# --- PDF Report Function ---
def generate_pdf_report(pdf_path, start_time, total_checks, duration_minutes, focus_retention, metrics, timeline_img_path):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    elements = []

    elements.append(Paragraph("Proctoring Session Report", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Start Time: {datetime.datetime.fromtimestamp(start_time)}", styles['Normal']))
    elements.append(Paragraph(f"End Time: {datetime.datetime.now()}", styles['Normal']))
    elements.append(Paragraph(f"Total Duration: {duration_minutes:.2f} minutes", styles['Normal']))
    elements.append(Paragraph(f"Total Checks: {total_checks}", styles['Normal']))
    elements.append(Paragraph(f"Times Looked Away (Face Missing): {metrics['looked_away_face_count']}", styles['Normal']))
    elements.append(Paragraph(f"Times Multiple People Detected: {metrics['multiple_faces_count']}", styles['Normal']))
    elements.append(Paragraph(f"Times Looked Away (Eye Gaze): {metrics['looked_away_eyes_count']}", styles['Normal']))
    elements.append(Paragraph(f"Focus Retention: {focus_retention:.2f}%", styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Violation Timestamps:", styles['Heading2']))
    for ts in metrics["timestamps"]:
        elements.append(Paragraph(ts, styles['Normal']))
    elements.append(Spacer(1, 12))

    elements.append(Paragraph("Timeline Visualization:", styles['Heading2']))
    elements.append(RLImage(timeline_img_path, width=500, height=50))

    doc.build(elements)

# --- Main Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, now_str, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    now = time.time()
    if now - last_check_time >= CHECK_INTERVAL:
        last_check_time = now
        total_checks += 1
        face_count = len(faces)

        if face_count == 1:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                if check_gaze_direction(face_landmarks.landmark):
                    current_state = "green"
                    append_timeline("green", CHECK_INTERVAL)
                else:
                    metrics["looked_away_eyes_count"] += 1
                    metrics["timestamps"].append(f"{datetime.datetime.now().strftime('%H:%M:%S')} - Eye Gaze Away")
                    current_state = "red"
                    append_timeline("red", CHECK_INTERVAL)
            else:
                metrics["looked_away_face_count"] += 1
                metrics["timestamps"].append(f"{datetime.datetime.now().strftime('%H:%M:%S')} - No Face Detected")
                current_state = "red"
                append_timeline("red", CHECK_INTERVAL)

        elif face_count == 0:
            metrics["looked_away_face_count"] += 1
            metrics["timestamps"].append(f"{datetime.datetime.now().strftime('%H:%M:%S')} - No Face Detected")
            current_state = "red"
            append_timeline("red", CHECK_INTERVAL)

        else:
            metrics["multiple_faces_count"] += 1
            metrics["timestamps"].append(f"{datetime.datetime.now().strftime('%H:%M:%S')} - Multiple Faces Detected")
            current_state = "red"
            append_timeline("red", CHECK_INTERVAL)

    cv2.imshow("Proctoring", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        metrics["timestamps"].append(f"{datetime.datetime.now().strftime('%H:%M:%S')} - Test Ended")
        break

cap.release()
cv2.destroyAllWindows()

# --- Metrics ---
total_time = time.time() - start_time
green_time = sum(d for color, d in timeline if color == "green")
focus_retention = (green_time / total_time) * 100
duration_minutes = total_time / 60

# --- Timeline Plot ---
fig, ax = plt.subplots(figsize=(10, 1))
start_pos = 0
for color, duration in timeline:
    ax.barh(0, duration, left=start_pos, color=color)
    start_pos += duration
ax.set_xlim(0, total_time)
ax.set_yticks([])
ax.set_xlabel("Time (seconds)")
timeline_img_path = "timeline.png"
plt.savefig(timeline_img_path, bbox_inches='tight')
plt.close()

# --- Generate PDF ---
pdf_path = "Proctoring_Report.pdf"
generate_pdf_report(pdf_path, start_time, total_checks, duration_minutes, focus_retention, metrics, timeline_img_path)
os.remove(timeline_img_path)

print(f"\nReport saved as {pdf_path}")
print("--- Test Metrics ---")
print(f"Total test duration: {duration_minutes:.2f} minutes")
print(f"Total checks: {total_checks}")
print(f"Times looked away (Face Missing): {metrics['looked_away_face_count']}")
print(f"Times multiple people detected: {metrics['multiple_faces_count']}")
print(f"Times looked away (Eye Gaze): {metrics['looked_away_eyes_count']}")
print(f"Focus retention: {focus_retention:.2f}%")
