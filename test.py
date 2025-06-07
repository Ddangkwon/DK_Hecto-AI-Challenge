import os
from glob import glob
import json

import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing import image

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

results = []

# 예측 반복
for path in tqdm(image_paths, desc="이미지 예측 중"):
    img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0

    pred = model.predict(x, verbose=0)[0]
    one_hot = np.zeros_like(pred)
    one_hot[np.argmax(pred)] = 1.0

    results.append([os.path.splitext(os.path.basename(path))[0]] + one_hot.tolist())


# 결과 저장
df = pd.DataFrame(results, columns=["ID"] + class_list)
df.to_csv("submission.csv", index=False, encoding='utf-8-sig')  # 엑셀 한글 호환을 위해 utf-8-sig 사용
