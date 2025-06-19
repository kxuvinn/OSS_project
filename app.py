from flask import Flask, render_template, request, jsonify
import torch
import cv2
import numpy as np
import sys
from pathlib import Path

app = Flask(__name__)

# yolov5 로컬 경로를 sys.path에 추가
sys.path.append(str(Path(__file__).resolve().parent / 'yolov5'))

# yolov5 내부 모듈 import
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.augmentations import letterbox

# 모델 불러오기
device = 'cpu'  # GPU 쓰려면 'cuda'
model = DetectMultiBackend('model/korean_model.pt', device=device)
model.eval()

model.names = [
    'giyeok', 'nieun', 'digeut', 'rieul', 'mieum', 'bieup', 'siot', 'ieung',
    'jieut', 'chieut', 'kieuk', 'tieut', 'pieup', 'hieut',
    'a', 'ya', 'eo', 'yeo', 'o', 'yo', 'u', 'yu', 'eu', 'i',
    'ae', 'yae', 'e', 'ye', 'oe', 'wi', 'ui'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/quiz/korean')
def quiz_korean():
    return render_template('quiz_korean.html')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    target = request.form.get('target')

    # 영어 → 한글 라벨 매핑
    label_map = {
        'giyeok': 'ㄱ', 'nieun': 'ㄴ', 'digeut': 'ㄷ', 'rieul': 'ㄹ',
        'mieum': 'ㅁ', 'bieup': 'ㅂ', 'siot': 'ㅅ', 'ieung': 'ㅇ',
        'jieut': 'ㅈ', 'chieut': 'ㅊ', 'kieuk': 'ㅋ', 'tieut': 'ㅌ',
        'pieup': 'ㅍ', 'hieut': 'ㅎ', 'a': 'ㅏ', 'ya': 'ㅑ',
        'eo': 'ㅓ', 'yeo': 'ㅕ', 'o': 'ㅗ', 'yo': 'ㅛ',
        'u': 'ㅜ', 'yu': 'ㅠ', 'eu': 'ㅡ', 'i': 'ㅣ',
        'ae': 'ㅐ', 'yae': 'ㅒ', 'e': 'ㅔ', 'ye': 'ㅖ',
        'oe': 'ㅚ', 'wi': 'ㅟ', 'ui': 'ㅢ',
    }

    # 이미지 디코딩
    img0 = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # 전처리: resize + padding (letterbox)
    img = letterbox(img0, new_shape=640)[0]
    img = img.transpose((2, 0, 1))[::-1]  # BGR → RGB, HWC → CHW
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device).float() / 255.0  # 0~1 정규화
    if img.ndimension() == 3:
        img = img.unsqueeze(0)  # batch dim 추가

    # 예측
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred)[0]

    if pred is not None and len(pred):
        pred = pred.cpu()
        box = max(pred, key=lambda x: x[4])
        cls_id = int(box[5].item())
        conf = float(box[4].item())
        eng_label = model.names[cls_id]
        label = label_map.get(eng_label, 'unknown')
        is_correct = (label == target)
        
        print("✅ 모델 클래스 목록:", model.names)
        print("📂 model.names:", model.names)
        print("📦 pred tensor:", pred)
        print("🔍 예측 클래스:", eng_label)
        print("📌 confidence:", conf)
        print("🎯 target:", target)
        print("✅ 매칭 결과:", is_correct)
        print("🖼️ 입력 이미지 shape:", img.shape)
        print("📸 max pixel:", img.max().item(), "min pixel:", img.min().item())
        
    else:
        label = 'none'
        conf = 0.0
        is_correct = False

    return jsonify({
        'prediction': label,
        'confidence': round(conf, 3),
        'is_correct': is_correct,
        'target': target
    })

if __name__ == '__main__':
    app.run(debug=True)
