# 필수 라이브러리 임포트
import os
import threading
import subprocess
import time

import tensorflow as tf
from tensorflow.python.client import device_lib  # 로컬 디바이스 정보 출력용
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # 이미지 전처리 및 증강
from tensorflow.keras.applications import ResNet50  # 사전 학습된 ResNet50 불러오기
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D  # 레이어 구성 요소
from tensorflow.keras.optimizers import Adam  # 옵티마이저
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
# -----------------------------
# GPU 설정 및 확인
# -----------------------------

# 사용 가능한 GPU 수 확인
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


# RTX 3050처럼 VRAM이 4~8GB인 경우 메모리 부족 문제 방지용 설정
# -> 필요한 만큼 메모리를 점진적으로 할당하도록 설정
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 구체적인 디바이스 정보 출력 (GPU 이름, 메모리 등)
print("Specific Resource Information :", device_lib.list_local_devices())



def monitor_gpu(interval=600):
    """ 일정 시간 간격으로 GPU 사용 상태를 출력하는 함수 """
    while True:
        result = subprocess.run(['nvidia-smi'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print(result.stdout)
        time.sleep(interval)

# GPU 모니터링 쓰레드 실행
monitor_thread = threading.Thread(target=monitor_gpu, args=(10,), daemon=True)
monitor_thread.start()

# -----------------------------
# 학습 설정
# -----------------------------

# 입력 이미지 크기 (ResNet50은 기본적으로 224x224 이미지 입력 요구)
input_size = (224, 224)

# 학습 시 한 번에 처리할 이미지 수
batch_size = 32

# 전체 학습 반복 횟수 (epoch)
epochs = 1

# 학습/검증 데이터 경로 설정
train_dir = "open/train"  # 클래스별 폴더가 포함된 학습 이미지 디렉토리
val_dir = "open/test"     # 검증 이미지 디렉토리

# -----------------------------
# 데이터 전처리 및 증강
# -----------------------------

# 학습 데이터: 정규화 + 다양한 증강 (회전, 이동, 확대, 좌우반전 등)
train_datagen = ImageDataGenerator(
    rescale=1./255,              # 픽셀 값을 [0,1] 범위로 정규화
    rotation_range=15,           # 최대 ±15도 회전
    zoom_range=0.2,              # 20%까지 확대
    width_shift_range=0.1,       # 좌우로 10% 이동
    height_shift_range=0.1,      # 상하로 10% 이동
    horizontal_flip=True         # 좌우 반전
)

# 검증 데이터: 정규화만 적용 (증강 X)
val_datagen = ImageDataGenerator(rescale=1./255)

# 학습용 이미지 제너레이터: 클래스별 폴더에서 자동 분류
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=input_size,      # 모든 이미지를 동일 크기로 리사이즈
    batch_size=batch_size,
    class_mode='categorical'     # 다중 클래스 분류 → 원-핫 인코딩
)

# 검증용 이미지 제너레이터
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=input_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# -----------------------------
# 모델 정의 (ResNet50 기반 전이 학습)
# -----------------------------
MODEL_PATH = 'car_model_classifier.h5'

# 모델 로딩 또는 새 모델 생성
if os.path.exists(MODEL_PATH):
    print(f"기존 모델 로드: {MODEL_PATH}")
    model = load_model(MODEL_PATH)
else:
    # 사전 학습된 ResNet50을 feature extractor로 사용 (분류기 제거)
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # ResNet50의 출력 feature map을 global average pooling (벡터로 축소)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    # Dense Layer 추가 (256개의 뉴런, ReLU 활성화 함수)
    x = Dense(256, activation='relu')(x)

    # 최종 출력층: 클래스 수 만큼의 뉴런, 소프트맥스로 확률 출력
    num_classes = len(train_generator.class_indices)
    predictions = Dense(num_classes, activation='softmax')(x)

    # 모델 구성 (입력 → ResNet50 → GAP → Dense → softmax 출력)
    model = Model(inputs=base_model.input, outputs=predictions)

# -----------------------------
# 모델 컴파일 및 학습
# -----------------------------

# 손실 함수: categorical_crossentropy (다중 클래스 분류용)
# 옵티마이저: Adam (학습률 1e-4로 설정)
# 평가지표: accuracy
model.compile(
    optimizer=Adam(1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 학습 시작: 학습 데이터와 검증 데이터 사용
checkpoint_cb = ModelCheckpoint(
    filepath='checkpoints/best_model_epoch_{epoch:02d}.h5',
    monitor='val_accuracy',
    save_best_only=False,
    verbose=1
)

model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[checkpoint_cb]
)

# -----------------------------
# 모델 저장
# -----------------------------
model.save('car_model_classifier.h5')
print("모델이 car_model_classifier.h5 파일로 저장되었습니다.")

# -----------------------------
# 클래스 인덱스 저장 (예측 시 라벨명 복원용)
# -----------------------------

# 예: {'BMW_1series': 0, 'Avante_AD': 1, ...}
# 예측 시 인덱스를 → 라벨명으로 변환하기 위해 저장해둠
import json
with open("class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)
