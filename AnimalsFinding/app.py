import torch
import cv2

# COCOにおける動物ラベ ル
animal_labels = {"bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}

# YOLOv5 モデルをロード（CPU）
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to('cpu')
model.eval()

# カメラ起動
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("カメラを開けませんでした")
    exit()

already_detected = set()

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
            print(f"🔥 Detected a new animal!: {label_en}")
            already_detected.add(label_en)

        # バウンディングボックスも描く
        x1, y1, x2, y2 = map(int, xyxy)
        color = (0, 255, 0) if label_en in animal_labels else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label_en} {conf_val:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    # 消えてしまったものを解除
    disappeared = already_detected - current_detected
    already_detected -= disappeared

    # === 検出リストを半透明ボックス付きで表示 ===
    overlay = frame.copy()
    list_x, list_y = 10, 30
    box_width = 200
    line_height = 25
    list_height = (len(already_detected) + 1) * line_height + 10

    # 半透明の黒背景（左上に）
    cv2.rectangle(overlay, (list_x - 5, list_y - 20),
                   (list_x + box_width, list_y - 20 + list_height),
                   (0, 0, 0), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # ラベルリスト描画（英語）
    cv2.putText(frame, "Detected:", (list_x, list_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    for i, label_en in enumerate(sorted(already_detected)):
        y = list_y + (i + 1) * line_height
        cv2.putText(frame, f"- {label_en}", (list_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 表示
    cv2.imshow("YOLOv5 Detection - with Filtered List", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
