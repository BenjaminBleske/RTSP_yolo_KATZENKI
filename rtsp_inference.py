from flask import Flask, Response, jsonify, render_template
import cv2
import time
from ultralytics import YOLO
from collections import defaultdict, deque


RTSP_URL = "rtsp://admin:KATZENKI199720192025@192.168.178.150:554/h264Preview_01_sub"

app = Flask(__name__)

model = YOLO('best.pt')  # oder yolov8s.pt falls schnell genug

recognition_log = defaultdict(deque)
last_seen = dict()
SECONDS_24H = 24 * 60 * 60

# Für letzte Einzelbilder + Galerie (max. 10)
last_detected_images = {
    'rocky_n': None,
    'rocky_t': None,
    'scratchy_n': None,
    'scratchy_t': None
}
gallery_images = defaultdict(lambda: deque(maxlen=10))

# Debounce für Bilder: Nur alle X Sekunden ein neues pro Klasse
last_save_time = {
    'rocky_n': 0,
    'rocky_t': 0,
    'scratchy_n': 0,
    'scratchy_t': 0
}
MIN_TIME_BETWEEN_SAVES = 10  # Sekunden


def gen_frames():
    while True:  # Äußere Schleife für automatischen Reconnect
        cap = cv2.VideoCapture(RTSP_URL)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Puffer klein halten

        if not cap.isOpened():
            print("Fehler: RTSP-Stream konnte nicht geöffnet werden! Warte 5 Sekunden...")
            time.sleep(5)
            continue

        print("RTSP-Stream erfolgreich verbunden.")

        last_inference = 0
        last_frame = None

        while True:
            success, frame = cap.read()
            if not success:
                print("Fehler beim Lesen eines Frames! Versuche reconnect...")
                cap.release()
                time.sleep(2)
                break  # Zur äußeren Schleife → neu verbinden

            current_time = time.time()

            # Inference nur alle 1 Sekunde
            if (current_time - last_inference) >= 1.0 or last_frame is None:
                results = model(frame)[0]

                # Frame für Stream vorbereiten (kopieren, falls Annotationen)
                annotated_frame = frame.copy()

                for box in results.boxes:
                    conf = float(box.conf[0])
                    
                    # Nur ab 75% (oder was du willst) weiterverarbeiten und anzeigen
                    if conf < 0.60:
                        continue  # Überspringen: weder loggen, speichern noch zeichnen

                    label = model.names[int(box.cls[0])]
                    now = time.time()

                    # Logging und Statistik
                    recognition_log[label].append(now)
                    last_seen[label] = now
                    while recognition_log[label] and recognition_log[label][0] < now - SECONDS_24H:
                        recognition_log[label].popleft()

                    # Galerie speichern (mit Debounce)
                    if label in last_detected_images:
                        if now - last_save_time[label] > MIN_TIME_BETWEEN_SAVES:
                            frame_copy = frame.copy()
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame_copy, f'{label} {conf:.2f}', (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            _, img_encoded = cv2.imencode('.jpg', frame_copy)
                            last_detected_images[label] = img_encoded.tobytes()
                            gallery_images[label].appendleft(img_encoded.tobytes())
                            last_save_time[label] = now

                    # Bounding Box und Label ins Live-Bild zeichnen
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                last_frame = annotated_frame
                last_inference = current_time


            # Streamen des letzten annotierten Frames
            if last_frame is not None:
                ret, buffer = cv2.imencode('.jpg', last_frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            time.sleep(0.04)  # ~25 FPS Stream

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
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

