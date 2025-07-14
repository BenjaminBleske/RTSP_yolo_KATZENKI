from flask import Flask, Response, jsonify
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
    return '''
    <!DOCTYPE html>
    <html lang="de">
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width,initial-scale=1" />
        <title>RTSP Stream mit YOLOv8</title>
        <style>
          body {
            font-family: 'Inter', Arial, sans-serif;
            background: #f6f8fa;
            margin: 0; padding: 0;
            color: #1a1a1a;
          }
          header {
            background: #222b45;
            color: #fff;
            padding: 24px 0 12px 0;
            text-align: center;
            border-bottom-left-radius: 32px;
            border-bottom-right-radius: 32px;
            box-shadow: 0 4px 16px rgba(60,70,100,0.06);
          }
          h1 {
            margin: 0;
            font-size: 2.4rem;
            font-weight: 700;
            letter-spacing: 1px;
          }
          .stream-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin-top: 32px;
          }
          .video-frame {
            background: #fff;
            border-radius: 24px;
            box-shadow: 0 2px 18px rgba(60,70,100,0.11);
            padding: 14px 14px 4px 14px;
            max-width: 90vw;
          }
          .video-frame img {
            border-radius: 18px;
            width: 720px;
            max-width: 100%;
            background: #111;
          }
          .stats-panel {
            margin: 36px auto 0 auto;
            background: #fff;
            border-radius: 16px;
            box-shadow: 0 2px 10px rgba(60,70,100,0.07);
            max-width: 420px;
            padding: 18px 24px 14px 24px;
            text-align: left;
          }
          .stats-panel h2 {
            font-size: 1.1rem;
            margin: 0 0 10px 0;
            color: #222b45;
            font-weight: 600;
          }
          #detections-panel {
            margin: 48px auto 0 auto;
            background: #f8fafc;
            border-radius: 14px;
            min-height: 90px;
            max-width: 700px;
            padding: 12px 22px;
            box-shadow: 0 1px 8px rgba(60,70,100,0.04);
          }
          .gallery-list {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
            justify-content: center;
            align-items: flex-start;
            margin-top: 10px;
          }
          .gallery-main-img {
            border-radius: 12px;
            box-shadow: 0 2px 10px #52b7c133;
            border: 2px solid #52b7c1;
            width: 180px;
            height: auto;
            max-width: 95vw;
          }
          .gallery-thumb {
            border-radius: 9px;
            width: 80px;
            height: auto;
            box-shadow: 0 1px 6px #999;
            margin-bottom: 3px;
          }
          @media (max-width: 800px) {
            .video-frame img { width: 98vw; }
            .stream-container { margin-top: 12px; }
            .stats-panel, #detections-panel { max-width: 96vw; }
            .gallery-main-img { width: 95vw; }
            .gallery-thumb { width: 23vw; }
          }
        </style>
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
      </head>
      <body>
        <header>
          <h1>Live-Objekterkennung (YOLOv8)</h1>
          <p style="margin: 8px 0 0 0; font-weight:400; font-size:1.12rem;">RTSP-Stream vom Raspberry Pi mit Echtzeit-Inferenz</p>
        </header>
        <div class="stream-container">
          <div class="video-frame">
            <img src="/video_feed" id="main-stream" alt="Live Video Stream" />
          </div>
        </div>
        <div class="stats-panel" id="stats-panel">
          <h2>Erkannte Objekte (letzte 24h)</h2>
          <div id="stats-table">Wird geladen…</div>
        </div>
        <div id="detections-panel">
            <div style="display:flex; gap:38px; justify-content:center; align-items:flex-start; flex-wrap:wrap;">
                <div style="text-align:center;">
                    <div style="font-weight:600; margin-bottom:7px;">Letzte 10 Car-Erkennungen</div>
                    <div id="car-gallery" class="gallery-list"></div>
                </div>
                <div style="text-align:center;">
                    <div style="font-weight:600; margin-bottom:7px;">Letzte 10 Dog-Erkennungen</div>
                    <div id="dog-gallery" class="gallery-list"></div>
                </div>
            </div>
        </div>
        <script>
          // Statistik regelmäßig aktualisieren:
          async function fetchStats() {
            try {
              const res = await fetch('/stats');
              const stats = await res.json();
              let html = '<table style="width:100%;border-collapse:collapse;font-size:1rem;"><tr><th align=left>Klasse</th><th align=right>Erkannte</th><th align=right>Zuletzt</th></tr>';
              const keys = Object.keys(stats);
              if(keys.length===0) html += '<tr><td colspan=3 style="padding:10px;text-align:center;color:#aaa;">Noch keine Erkennung</td></tr>';
              for (const [k, v] of Object.entries(stats)) {
                html += `<tr><td>${k}</td><td align=right>${v.count_24h}</td><td align=right>${v.last_seen??''}</td></tr>`;
              }
              html += '</table>';
              document.getElementById('stats-table').innerHTML = html;
            } catch (e) {
              document.getElementById('stats-table').innerHTML = 'Fehler beim Laden der Statistik.';
            }
          }
          fetchStats();
          setInterval(fetchStats, 3000);

          async function loadGallery(label) {
            const res = await fetch(`/gallery_count/${label}`);
            const {count} = await res.json();
            let html = '';
            const ts = Date.now();
            for(let i=0; i<count; ++i){
              if(i===0) {
                html += `<img src="/gallery_image/${label}/0?t=${ts}" class="gallery-main-img" alt="${label} #1">`;
              } else {
                html += `<img src="/gallery_image/${label}/${i}?t=${ts}" class="gallery-thumb" alt="${label} #${i+1}">`;
              }
            }
            document.getElementById(`${label}-gallery`).innerHTML = html || "<em style='color:#aaa;'>Keine Erkennungen</em>";
          }
          setInterval(() => {
              loadGallery('car');
              loadGallery('dog');
          }, 3000);
          loadGallery('car');
          loadGallery('dog');
        </script>
      </body>
    </html>
    '''

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
