import numpy as np
from keras.datasets import mnist
from maraboupy.MarabouNetworkONNX import MarabouNetworkONNX
from maraboupy import Marabou

import onnxruntime as rt  # 추가된 부분

# 1. MNIST 데이터 로드 및 전처리
(_, _), (test_images, test_labels) = mnist.load_data()
test_images = test_images / 255.0  # [0, 255] → [0, 1] 정규화

# 2. 5개 샘플 추출 (0-9 클래스 균형있게 선택)
sample_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # 첫 5개 샘플 선택
samples = test_images[sample_indices].reshape(10, 28, 28, 1)
labels = test_labels[sample_indices]

model_path = "/home/minyeong/realiable/mnist_cnn_trained.onnx"
network = MarabouNetworkONNX(model_path)

# 3. ONNX 모델 예측 검증용 세션 생성 (추가된 부분)
ort_session = rt.InferenceSession(model_path)
input_name = ort_session.get_inputs()[0].name
output_name = ort_session.get_outputs()[0].name

# 4. 견고성 검증 파라미터 설정
a = 0.0007 # 섭동 범위 (±5%)
timeout = 300  # 검증 제한 시간(초)

# 5. 각 샘플에 대해 검증 수행
for idx in range(10):
    # 입력 이미지 및 라벨 가져오기
    original_image = samples[idx].flatten()
    true_label = labels[idx]
    
    # 6-1. ONNX Runtime으로 원본 이미지 예측 확인 (추가된 부분)
    img_for_ort = samples[idx].reshape(1, 28, 28, 1).astype(np.float32)
    ort_pred = ort_session.run([output_name], {input_name: img_for_ort})[0]
    pred_label = np.argmax(ort_pred)
    
    print(f"\n샘플 {idx+1} 실제 라벨: {true_label}, 모델 예측: {pred_label}")
    
    # 6-2. 모델이 원본을 오분류하면 검증 생략
    if pred_label != true_label:
        print("→ 모델이 원본 이미지를 오분류함. 검증 불필요")
        continue

    # 입력 변수 경계 설정
    input_vars = network.inputVars[0]
    for i, var in enumerate(input_vars.flatten()):
        perturb_min = max(original_image[i] -a + 0.1, 0.0)

        perturb_max = min(original_image[i] +a +0.1, 1.0)
        print(perturb_max)
        network.setLowerBound(int(var), perturb_min)
        network.setUpperBound(int(var), perturb_max)
    # 출력 제약 조건 추가 (정답 클래스가 최대값 유지)
    output_vars = network.outputVars[0][0]
    for i in range(10):
        if i != true_label:
            network.addInequality([output_vars[true_label], output_vars[i]], [1, -1], 0)
    
    options = Marabou.createOptions(timeoutInSeconds=1000, verbosity=0)

    # Marabou 검증 실행
    result = network.solve(filename="", verbose=True, options=options)

    # 결과 판정
    if result[0] != 'unsat':
        print(f"샘플 {idx+1} (라벨 {true_label}): 취약 (SAT) → 적대적 예제 존재")
    else:
        print(f"샘플 {idx+1} (라벨 {true_label}): 안전 (UNSAT) → {a} 범위 내 보장")

