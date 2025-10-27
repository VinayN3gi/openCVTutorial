import cv2
import mediapipe as mp
import time
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
import os

# SETTINGS
GRACE_PERIOD = 2    # seconds before marking violation
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 700

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)

# Stats tracking
violations = []
start_time = time.time()
stats = {
    "total_frames": 0,
    "violations": 0,
    "total_time": 0,
    "fps": 0
}


# --- Draw rectangles around eyes ---
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


# --- Generate PDF report ---
def generate_pdf_report(stats, violations):
    report_path = "eye_violation_report.pdf"
    doc = SimpleDocTemplate(report_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("<b>Eye Tracking Violation Report</b>", styles['Title']))
    elements.append(Spacer(1, 20))

    # Summary Table
    summary_data = [
        ["Metric", "Value"],
        ["Total Time (s)", f"{stats['total_time']:.2f}"],
        ["Total Frames", stats['total_frames']],
        ["Average FPS", f"{stats['fps']:.2f}"],
        ["Total Violations", stats['violations']],
        ["Violation Rate (%)", f"{(stats['violations'] / max(stats['total_frames'], 1)) * 100:.2f}%"]
    ]
    summary_table = Table(summary_data, hAlign='LEFT')
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (1, 1), (-1, -1), 'RIGHT')
    ]))
    elements.append(summary_table)
    elements.append(Spacer(1, 20))

    # Violation Timeline
    if violations:
        times = [v[0] for v in violations]
        values = [1] * len(times)
        plt.figure(figsize=(6, 1))
        plt.scatter(times, values, c='red')
        plt.yticks([])
        plt.xlabel("Time (s)")
        plt.title("Violation Timeline")
        plt.tight_layout()
        timeline_img_path = "timeline_plot.png"
        plt.savefig(timeline_img_path)
        plt.close()

        elements.append(Paragraph("<b>Violation Timeline</b>", styles['Heading2']))
        elements.append(RLImage(timeline_img_path, width=400, height=100))
        elements.append(Spacer(1, 10))
        elements.append(Paragraph(f"Total Violations Recorded: {len(violations)}", styles['Normal']))
        doc.build(elements)
        os.remove(timeline_img_path)
    else:
        elements.append(Paragraph("No violations were detected.", styles['Normal']))
        doc.build(elements)

    print(f"[INFO] PDF report generated successfully → {report_path}")


# --- Main camera loop ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, WINDOW_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WINDOW_HEIGHT)

looking_away_start = None
status_text = "OK"
prev_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    current_time = time.time()
    stats["total_frames"] += 1

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Calculate FPS
    fps = 1.0 / (current_time - prev_time)
    prev_time = current_time
    stats["fps"] = (stats["fps"] * 0.9) + (fps * 0.1)  # smoothed FPS

    # Face tracking logic
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
        is_looking_away = True
        status_text = "No Face Detected"

    # Violation check
    if is_looking_away:
        if looking_away_start is None:
            looking_away_start = time.time()
        elif time.time() - looking_away_start > GRACE_PERIOD:
            stats["violations"] += 1
            violations.append((time.time() - start_time, status_text))
            looking_away_start = None
    else:
        looking_away_start = None

    elapsed = time.time() - start_time
    violation_rate = (stats["violations"] / max(stats["total_frames"], 1)) * 100

    # On-screen metrics
    cv2.putText(frame, f"STATUS: {status_text}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Violations: {stats['violations']}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(frame, f"Time: {elapsed:.1f}s", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
    cv2.putText(frame, f"FPS: {stats['fps']:.1f}", (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"Violation Rate: {violation_rate:.2f}%", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Eye Tracker - Live Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stats["total_time"] = time.time() - start_time
cap.release()
cv2.destroyAllWindows()

# Generate report
generate_pdf_report(stats, violations)
