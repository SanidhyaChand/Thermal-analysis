# app.py

import cv2
import numpy as np
import base64
from flask import Flask, render_template, request

# --- Initialize Flask App ---
app = Flask(__name__)


# --- ThermalImageAnalyzer Class ---
class ThermalImageAnalyzer:
    """Analyzes thermal images with user-defined temperature ranges and thresholds."""

    def __init__(self, temp_min=0.0, temp_max=100.0):
        self.TEMP_MIN = float(temp_min)
        self.TEMP_MAX = float(temp_max)
        self.HOTSPOT_THRESHOLD_INTENSITY = 255
        self.COLDSPOT_THRESHOLD_INTENSITY = 0
        self.MIN_CONTOUR_AREA = 50

    def temperature_to_intensity(self, temp):
        if self.TEMP_MAX == self.TEMP_MIN: return 128
        intensity = 255.0 * (temp - self.TEMP_MIN) / (self.TEMP_MAX - self.TEMP_MIN)
        return np.clip(intensity, 0, 255).astype(np.uint8)

    def intensity_to_temperature(self, intensity):
        if self.TEMP_MAX == self.TEMP_MIN: return self.TEMP_MIN
        return self.TEMP_MIN + (intensity / 255.0) * (self.TEMP_MAX - self.TEMP_MIN)

    def load_image_from_data(self, image_stream):
        data = np.frombuffer(image_stream.read(), np.uint8)
        image_16bit = cv2.imdecode(data, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE)
        if image_16bit is not None and image_16bit.dtype == np.uint16:
            image = cv2.convertScaleAbs(image_16bit, alpha=255.0 / 65535.0)
        else:
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if image is None: raise ValueError("Could not decode image from data stream.")
        return image

    def convert_to_thermal(self, image):
        if len(image.shape) == 3:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            grayscale = image.copy()
        thermal_image = cv2.applyColorMap(grayscale, cv2.COLORMAP_JET)
        return thermal_image, grayscale

    def detect_spots(self, grayscale):
        _, hotspot_mask = cv2.threshold(grayscale, self.HOTSPOT_THRESHOLD_INTENSITY, 255, cv2.THRESH_BINARY)
        _, coldspot_mask = cv2.threshold(grayscale, self.COLDSPOT_THRESHOLD_INTENSITY, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        hotspot_mask = cv2.morphologyEx(hotspot_mask, cv2.MORPH_CLOSE, kernel)
        hotspot_contours, _ = cv2.findContours(hotspot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        coldspot_mask = cv2.morphologyEx(coldspot_mask, cv2.MORPH_CLOSE, kernel)
        coldspot_contours, _ = cv2.findContours(coldspot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_hotspots = [cnt for cnt in hotspot_contours if cv2.contourArea(cnt) >= self.MIN_CONTOUR_AREA]
        valid_coldspots = [cnt for cnt in coldspot_contours if cv2.contourArea(cnt) >= self.MIN_CONTOUR_AREA]
        return valid_hotspots, valid_coldspots

    def analyze_regions(self, contours, grayscale, spot_type):
        region_data = []
        for i, contour in enumerate(contours):
            mask = np.zeros(grayscale.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(grayscale, mask=mask)
            mean_intensity = cv2.mean(grayscale, mask=mask)[0]
            avg_temp = self.intensity_to_temperature(mean_intensity)
            if spot_type == 'hot':
                peak_loc, peak_temp, spot_id = max_loc, self.intensity_to_temperature(max_val), f"H{i + 1}"
            else:
                peak_loc, peak_temp, spot_id = min_loc, self.intensity_to_temperature(min_val), f"C{i + 1}"
            region_data.append({'id': spot_id, 'contour': contour, 'bounding_rect': cv2.boundingRect(contour),
                                'peak_loc': peak_loc, 'avg_temp': avg_temp, 'peak_temp': peak_temp})
        return region_data

    def draw_annotations(self, image, analyzed_data, color):
        """Draws bounding boxes, IDs, and peak points on the image."""
        for data in analyzed_data:
            x, y, w, h = data['bounding_rect']
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 1)
            cv2.circle(image, data['peak_loc'], 5, (255, 255, 255), -1)
            # Put the ID text inside the box using the rectangle's color
            cv2.putText(image, data['id'], (x + 5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)  # Changed color from white
        return image


# --- Flask Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    # Set default context for GET requests
    context = {'highest_temp': 100, 'lowest_temp': 0, 'hotspot_temp': 65, 'coldspot_temp': 20, 'display_mode': 'both'}

    if request.method == 'POST':
        file = request.files.get('file')
        # Update context with form data
        context['highest_temp'] = request.form.get('highest_temp', 100)
        context['lowest_temp'] = request.form.get('lowest_temp', 0)
        context['hotspot_temp'] = request.form.get('hotspot_temp', 65)
        context['coldspot_temp'] = request.form.get('coldspot_temp', 20)
        context['display_mode'] = request.form.get('display_mode', 'both')

        if not file or file.filename == '':
            context['error'] = 'No file selected'
            return render_template('index.html', **context)

        try:
            analyzer = ThermalImageAnalyzer(temp_min=context['lowest_temp'], temp_max=context['highest_temp'])
            analyzer.HOTSPOT_THRESHOLD_INTENSITY = analyzer.temperature_to_intensity(float(context['hotspot_temp']))
            analyzer.COLDSPOT_THRESHOLD_INTENSITY = analyzer.temperature_to_intensity(float(context['coldspot_temp']))

            image = analyzer.load_image_from_data(file.stream)
            thermal_image, grayscale = analyzer.convert_to_thermal(image)
            hot_contours, cold_contours = analyzer.detect_spots(grayscale)

            hotspot_data = []
            coldspot_data = []

            if context['display_mode'] in ['both', 'hot_only']:
                hotspot_data = analyzer.analyze_regions(hot_contours, grayscale, 'hot')

            if context['display_mode'] in ['both', 'cold_only']:
                coldspot_data = analyzer.analyze_regions(cold_contours, grayscale, 'cold')

            annotated_image = thermal_image.copy()

            if hotspot_data:
                annotated_image = analyzer.draw_annotations(annotated_image, hotspot_data, color=(0, 0, 255))
            if coldspot_data:
                annotated_image = analyzer.draw_annotations(annotated_image, coldspot_data, color=(255, 0, 0))

            _, buffer = cv2.imencode('.jpg', annotated_image)

            context['processed_image'] = base64.b64encode(buffer).decode('utf-8')
            context['hotspot_report'] = hotspot_data
            context['coldspot_report'] = coldspot_data

        except Exception as e:
            context['error'] = f"An error occurred: {e}"

        return render_template('index.html', **context)

    return render_template('index.html', **context)


if __name__ == '__main__':
    app.run(debug=True)