import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
import os


# SETTINGS
GRACE_PERIOD = 2    # seconds before marking violation
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Track violations and stats
violations = []
start_time = time.time()
stats = {
    "total_frames": 0,
    "violations": 0,
    "total_time": 0
}


# --- Function to draw rectangles around eyes ---
def draw_eye_boxes(frame, landmarks):
    h, w, _ = frame.shape

    # Left Eye
    left_x_min = int(landmarks[33].x * w)
    left_x_max = int(landmarks[133].x * w)
    left_y_min = int(min(landmarks[i].y for i in [33, 159, 145]) * h)
    left_y_max = int(max(landmarks[i].y for i in [133, 23, 27]) * h)
    cv2.rectangle(frame, (left_x_min, left_y_min), (left_x_max, left_y_max), (0, 255, 0), 2)

    # Right Eye
    right_x_min = int(landmarks[362].x * w)
    right_x_max = int(landmarks[263].x * w)
    right_y_min = int(min(landmarks[i].y for i in [362, 386, 374]) * h)
    right_y_max = int(max(landmarks[i].y for i in [263, 253, 249]) * h)
    cv2.rectangle(frame, (right_x_min, right_y_min), (right_x_max, right_y_max), (0, 255, 0), 2)


# --- Function to generate PDF report ---
def generate_pdf_report(stats, violations):
    report_path = "violation_report_read5.pdf" # Renamed to avoid overwriting other reports
    doc = SimpleDocTemplate(report_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Eye Tracking Violation Report", styles['Title']))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Total Time: {stats['total_time']:.2f} seconds", styles['Normal']))
    elements.append(Paragraph(f"Total Frames Processed: {stats['total_frames']}", styles['Normal']))
    elements.append(Paragraph(f"Total Violations: {stats['violations']}", styles['Normal']))
    elements.append(Spacer(1, 12))

    if violations:
        # Create timeline plot
        times = [v[0] for v in violations]
        values = [1] * len(times)

        plt.figure(figsize=(6, 1))
        plt.scatter(times, values, c='red')
        plt.yticks([])
        plt.xlabel("Time (s)")
        plt.title("Violation Timeline")
        timeline_img_path = "timeline.png"
        plt.savefig(timeline_img_path)
        plt.close()

        elements.append(Paragraph("Violation Timeline:", styles['Heading2']))
        elements.append(RLImage(timeline_img_path, width=400, height=100))
        doc.build(elements)

        if os.path.exists(timeline_img_path):
            os.remove(timeline_img_path)
    else:
        elements.append(Paragraph("No violations were recorded.", styles['Normal']))
        doc.build(elements)


    print(f"[INFO] PDF report generated: {report_path}")


# --- Main Camera Loop ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

looking_away_start = None
status_text = "OK"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # --- FIX: Flip the frame horizontally ---
    frame = cv2.flip(frame, 1)

    stats["total_frames"] += 1
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # --- REVISED VIOLATION LOGIC ---
    is_looking_away = False
    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        draw_eye_boxes(frame, landmarks)

        nose_x = landmarks[1].x
        if nose_x < 0.3 or nose_x > 0.7:
            is_looking_away = True
            status_text = "Looking Away"
        else:
            status_text = "Centered"
    else:
        # Treat "no face" as a looking away event
        is_looking_away = True
        status_text = "No Face Detected"

    # --- Timer and Violation Counter ---
    if is_looking_away:
        if looking_away_start is None:
            looking_away_start = time.time()
        elif time.time() - looking_away_start > GRACE_PERIOD:
            stats["violations"] += 1
            violations.append((time.time() - start_time, status_text))
            looking_away_start = None
    else:
        looking_away_start = None

    # Display status on the frame
    cv2.putText(frame, f"STATUS: {status_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Violations: {stats['violations']}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Eye Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stats["total_time"] = time.time() - start_time
cap.release()
cv2.destroyAllWindows()

# Generate PDF report
generate_pdf_report(stats,(violations))