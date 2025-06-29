from flask import Flask, Response, render_template
import torch
import cv2
import requests
from db import Position

res = requests.get("https://ipinfo.io/json")
data = res.json()

app = Flask(__name__)

animal_labels = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}

# YOLOv5モデルロード
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to('cpu')
model.eval()

already_detected = set()

def generate_frames():
    global already_detected
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラを開けません")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)

        current_detected = set()

        for *xyxy, conf, cls in results.xyxy[0]:
            conf_val = float(conf)
            label_en = model.names[int(cls)]

            if conf_val < 0.2:
                continue

            current_detected.add(label_en)

            if label_en in animal_labels and label_en not in already_detected:
                print(f"Detected a new animal!: {label_en}")
                loc = data['loc'].split(',')


                latitude = float(loc[0])
                longitude = float(loc[1])                  

                Latitude  = latitude
                Longitude = longitude
                Position.create(Latitude=Latitude, Longitude=Longitude)


                already_detected.add(label_en)

            # 四角形描画
            x1, y1, x2, y2 = map(int, xyxy)
            color = (0, 255, 0) if label_en in animal_labels else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label_en} {conf_val:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # 消えたものはリストから削除
        disappeared = already_detected - current_detected
        already_detected -= disappeared

        # 半透明ボックスで検出リスト表示
        overlay = frame.copy()
        list_x, list_y = 10, 30
        box_width = 200
        line_height = 25
        list_height = (len(already_detected) + 1) * line_height + 10

        cv2.rectangle(overlay, (list_x - 5, list_y - 20),
                      (list_x + box_width, list_y - 20 + list_height),
                      (0, 0, 0), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        cv2.putText(frame, "Detected:", (list_x, list_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for i, label_en in enumerate(sorted(already_detected)):
            y = list_y + (i + 1) * line_height
            cv2.putText(frame, f"- {label_en}", (list_x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # JPEGエンコード
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # フレームをHTTPレスポンス形式でyield
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/list')
def list():
    Positions = Position.select()
    return render_template('list.html',Positions=Positions)

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
