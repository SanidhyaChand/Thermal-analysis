import cv2
import numpy as np
import base64
import sqlite3
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)

# Load YOLO Model
try:
    yolo_model = YOLO('best.pt')
except Exception as e:
    print(f"Error loading model: {e}")


class ThermalDiagnostics:
    def __init__(self, t_min, t_max):
        self.T_MIN = float(t_min)
        self.T_MAX = float(t_max)

    def get_thresholds(self, detected_classes):
        """Fetches limits for ALL detected components"""
        if not detected_classes: return {}
        try:
            conn = sqlite3.connect('maintenance.db')
            cursor = conn.cursor()
            placeholders = ', '.join(['?'] * len(detected_classes))
            query = f"SELECT name, max_temp_c FROM components WHERE name IN ({placeholders})"
            cursor.execute(query, list(detected_classes))
            results = {row[0]: row[1] for row in cursor.fetchall()}
            conn.close()
            return results
        except:
            return {}

    def intensity_to_temp(self, intensity):
        return self.T_MIN + (intensity / 255.0) * (self.T_MAX - self.T_MIN)

    def run_diagnostics(self, file_stream, manual_threshold=None):
        data = np.frombuffer(file_stream.read(), np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)

        # 1. AI IDENTIFICATION (Detection of multiple elements)
        results = yolo_model(img, imgsz=512, conf=0.5)[0]
        detected_info = []
        for box in results.boxes:
            cls_name = results.names[int(box.cls[0])]
            conf = float(box.conf[0])
            detected_info.append({'name': cls_name, 'conf': conf})
            # Visual Bounding Boxes
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{cls_name} {conf:.2f}", (x1, y1 - 10), 0, 0.5, (0, 255, 0), 2)

        is_low_confidence = len(results.boxes) == 0

        # 2. RESOLVE MULTIPLE THRESHOLDS
        safety_limits = self.get_thresholds([d['name'] for d in detected_info])
        # Use most restrictive (lowest) limit found, or fallback to manual input
        active_limit = min(safety_limits.values()) if safety_limits else float(manual_threshold or 65.0)
        caution_limit = active_limit * 0.85  # Middle ground

        # 3. 6x6 GRID ANALYSIS
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        ch, cw = h // 6, w // 6
        grid_report = []
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

        hazard_count, caution_count = 0, 0

        for i in range(6):
            for j in range(6):
                roi = gray[i * ch:(i + 1) * ch, j * cw:(j + 1) * cw]
                _, mv, _, ml = cv2.minMaxLoc(roi)
                temp = self.intensity_to_temp(mv)

                status = "NORMAL"
                if temp >= active_limit:
                    status = "HAZARD"
                    hazard_count += 1
                    cv2.circle(thermal, (j * cw + ml[0], i * ch + ml[1]), 5, (0, 0, 255), -1)
                elif temp >= caution_limit:
                    status = "CAUTION"
                    caution_count += 1
                    cv2.circle(thermal, (j * cw + ml[0], i * ch + ml[1]), 5, (0, 255, 255), -1)

                cv2.rectangle(thermal, (j * cw, i * ch), ((j + 1) * cw, (i + 1) * ch), (255, 255, 255), 1)
                grid_report.append({'id': (i * 6) + j + 1, 'temp': f"{temp:.1f}", 'status': status})

        # 4. OVERALL HEALTH LOGIC (Area Ratios)
        h_ratio = (hazard_count / 36) * 100
        c_ratio = (caution_count / 36) * 100

        if h_ratio > 10 or c_ratio > 40:
            flag = "🔴 SYSTEM HAZARD"
        elif caution_count > 0:
            flag = "🟡 SYSTEM CAUTION"
        else:
            flag = "🟢 SYSTEM SAFE"

        ratio_text = f"Status: Hazard {h_ratio:.1f}% | Caution {c_ratio:.1f}% | Normal {(100 - h_ratio - c_ratio):.1f}%"

        # Return all 7 values to fix the ValueError
        return thermal, grid_report, detected_info, active_limit, is_low_confidence, flag, ratio_text


@app.route('/', methods=['GET', 'POST'])
def index():
    ctx = {'t_max': 100, 't_min': 0, 'manual_thresh': 65}
    if request.method == 'POST' and request.files.get('file'):
        proc = ThermalDiagnostics(request.form['t_min'], request.form['t_max'])
        # FIX: Catch all 7 returned values
        img, report, ai_data, limit, fallback, flag, ratio = proc.run_diagnostics(request.files['file'],
                                                                                  request.form['manual_thresh'])

        _, buf = cv2.imencode('.jpg', img)
        ctx.update({
            'img': base64.b64encode(buf).decode(),
            'report': report, 'ai_results': ai_data,
            'active_limit': limit, 'show_fallback': fallback,
            'overall_flag': flag, 'ratio_text': ratio
        })
    return render_template('index.html', **ctx)


if __name__ == '__main__':
    import os

    # Render provides the 'PORT' environment variable automatically
    # We default to 10000 if it's not found (Render's common default)
    port = int(os.environ.get("PORT", 10000))

    # Host must be '0.0.0.0' to be accessible externally
    app.run(host='0.0.0.0', port=port)
