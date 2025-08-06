import os
from glob import glob
import json

import pandas as pd
import numpy as np
import tensorflow as tf

# 모델 로드
model = tf.keras.models.load_model("car_model_classifier.h5")

# 클래스 인덱스 로드
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

index_to_class = {v: k for k, v in class_indices.items()}
class_list = [index_to_class[i] for i in range(len(index_to_class))]  # 정렬된 클래스 리스트

# 테스트 이미지 경로 설정 (test 하위에 이미지 파일만 있는 구조)
TEST_DIR = "open/test"
image_paths = sorted(glob(os.path.join(TEST_DIR, "*.jpg")))  # 정렬은 ID 순서 유지 위함

# 데이터셋 생성 및 성능 향상을 위한 prefetch 적용
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    TEST_DIR,
    labels=None,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
).map(lambda x: x / 255.0).prefetch(tf.data.AUTOTUNE)

# 배치 단위 예측으로 속도 향상
preds = model.predict(test_dataset, verbose=0)
max_indices = np.argmax(preds, axis=1)
one_hot = np.zeros_like(preds)
one_hot[np.arange(len(preds)), max_indices] = 1.0

# ID와 예측 결과 매핑
results = [
    [os.path.splitext(os.path.basename(path))[0]] + one_hot[i].tolist()
    for i, path in enumerate(image_paths)
]

# 결과 저장
df = pd.DataFrame(results, columns=["ID"] + class_list)
df.to_csv("submission.csv", index=False, encoding='utf-8-sig')  # 엑셀 한글 호환을 위해 utf-8-sig 사용
