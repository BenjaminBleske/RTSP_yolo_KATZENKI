from flask import Flask, Response, jsonify, render_template
import cv2
import time
from ultralytics import YOLO
from collections import defaultdict, deque


RTSP_URL = "rtsp://admin:KATZENKI199720192025@192.168.178.150:554/h264Preview_01_sub"

app = Flask(__name__)

model = YOLO('yolov8n.pt')  # oder yolov8s.pt falls schnell genug

recognition_log = defaultdict(deque)
last_seen = dict()
SECONDS_24H = 24 * 60 * 60

# Für letzte Einzelbilder + Galerie (max. 10)
last_detected_images = {'car': None, 'dog': None}
gallery_images = defaultdict(lambda: deque(maxlen=10))

# Debounce für Bilder: Nur alle X Sekunden ein neues pro Klasse
last_save_time = {'car': 0, 'dog': 0}
MIN_TIME_BETWEEN_SAVES = 10  # Sekunden

def gen_frames():
    cap = cv2.VideoCapture(RTSP_URL)
    if not cap.isOpened():
        print("Fehler: RTSP-Stream konnte nicht geöffnet werden!")
        return

    last_inference = 0
    last_frame = None

    while True:
        success, frame = cap.read()
        if not success:
            print("Fehler beim Lesen eines Frames!")
            time.sleep(0.2)
            continue

        current_time = time.time()
        # Nur alle 1 Sekunde Inference
        if (current_time - last_inference) >= 1.0 or last_frame is None:
            results = model(frame)[0]

            for box in results.boxes:
                label = model.names[int(box.cls[0])]
                now = time.time()

                recognition_log[label].append(now)
                last_seen[label] = now

                while recognition_log[label] and recognition_log[label][0] < now - SECONDS_24H:
                    recognition_log[label].popleft()

                # Nur speichern, wenn die letzte Speicherung lange genug her ist
                if label in last_detected_images:
                    if now - last_save_time[label] > MIN_TIME_BETWEEN_SAVES:
                        # Bounding Box zeichnen (nur auf Bild für Galerie)
                        frame_copy = frame.copy()
                        x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
                        conf = box.conf[0]
                        cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(frame_copy, f'{label} {conf:.2f}', (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                        _, img_encoded = cv2.imencode('.jpg', frame_copy)
                        last_detected_images[label] = img_encoded.tobytes()
                        gallery_images[label].appendleft(img_encoded.tobytes())
                        last_save_time[label] = now

                # Für Stream Bounding Box immer zeichnen (auf "frame")
                x1, y1, x2, y2 = [int(c) for c in box.xyxy[0]]
                conf = box.conf[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f'{label} {conf:.2f}', (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            last_frame = frame.copy()
            last_inference = current_time

        ret, buffer = cv2.imencode('.jpg', last_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        time.sleep(0.04)  # flüssige Anzeige im Browser

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/')
def index():
    return render_template('index.html', foo="bar")

@app.route('/gallery_image/<label>/<int:idx>')
def gallery_image(label, idx):
    imgs = gallery_images.get(label, [])
    if 0 <= idx < len(imgs):
        return Response(imgs[idx], mimetype='image/jpeg')
    else:
        from flask import abort
        abort(404)

@app.route('/gallery_count/<label>')
def gallery_count(label):
    return jsonify({'count': len(gallery_images[label])})

@app.route('/stats')
def stats():
    now = time.time()
    stats_dict = {}
    for label, times in recognition_log.items():
        count_last_24h = sum(1 for t in times if t > now - SECONDS_24H)
        last_time = last_seen.get(label, None)
        stats_dict[label] = {
            "count_24h": count_last_24h,
            "last_seen": time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_time)) if last_time else None
        }
    return jsonify(stats_dict)

@app.route('/download_image/<label>/<int:idx>')
def download_image(label, idx):
    imgs = gallery_images.get(label, [])
    if 0 <= idx < len(imgs):
        from flask import send_file
        import io
        return send_file(
            io.BytesIO(imgs[idx]),
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f"{label}_{idx+1}.jpg"
        )
    else:
        from flask import abort
        abort(404)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
