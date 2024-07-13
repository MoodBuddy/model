from flask import Flask, request, render_template, jsonify
import torch
import boto3
import os
import numpy as np
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification

app = Flask(__name__)

# KoBERT 모델과 토크나이저 로드
model_path = 'model_01.pt'

# # S3에서 모델 파일 다운로드
# s3 = boto3.client('s3')
# bucket_name = 'dylee-model'
# model_key = 'model_01.pt'
# model_path = '/tmp/model_01.pt'

# if not os.path.exists(model_path):
#     s3.download_file(bucket_name, model_key, model_path)
tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
model = BertForSequenceClassification.from_pretrained('monologg/kobert',num_labels=7)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
print("모델 로드 완료")
model.eval()  # 모델을 평가 모드로 전환

# emotion_dict = {0: "공포가", 1: "놀람이", 2: "분노가", 3: "슬픔이", 4: "중립이", 5: "행복이", 6: "혐오가"}
emotion_dict = {0: "FEAR", 1: "SURPRISE", 2: "ANGER", 3: "SADNESS", 4: "NEUTRAL", 5: "HAPPINESS", 6: "DISGUST"}

@app.route('/model', methods=['POST'])
def receive_string():
    # Spring으로부터 JSON 객체를 전달받음
    try:
        dto_json = request.get_json(force=True)
        if not dto_json:
            return jsonify({"error": "Empty JSON received"}), 400

        diary_summary = dto_json.get('diarySummary')
        if not diary_summary:
            return jsonify({"error": "No diarySummary field in JSON"}), 400

        inputs = tokenizer(diary_summary, return_tensors='pt')

        # 모델 예측
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # 소프트맥스 함수로 감정 확률 계산
        probabilities = torch.nn.functional.softmax(logits, dim=-1)

        # 가장 높은 확률의 감정 인덱스 추출
        predicted_class = torch.argmax(probabilities, dim=-1).item()

        # 결과 출력
        predicted_emotion = emotion_dict[predicted_class]
        print(f"입력 문장: '{diary_summary}'")
        print(f"예측 감정: {predicted_emotion}")

        # 결과를 JSON 형태로 반환
        response = {
            "emotion": predicted_emotion
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run('0.0.0.0', port=8099, debug=True)

