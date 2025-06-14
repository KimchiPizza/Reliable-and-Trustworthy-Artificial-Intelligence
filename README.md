# Marabou를 활용한 MNIST 신경망 강건성 검증

## 1. 개요
- 본 프로젝트는 Marabou를 활용하여 직접 학습한 MNIST CNN 모델의 강건성을 검증하는 실험입니다.
- 모델은 TensorFlow/Keras로 학습 후 ONNX로 변환하였으며, Marabou에서 입력 섭동에 대한 안전성을 수학적으로 검증합니다.

## 2. 파일 구성
- `train.py`: MNIST CNN 모델 학습 및 저장 (Keras H5)
- `keras2onnx.py`: 학습된 모델을 ONNX로 변환
- `main.py`: Marabou를 활용한 강건성 검증 (10개 샘플, 입력 섭동 ±0.0007)

## 3. 실행 방법
1. `train.py` 실행 → `mnist_cnn_trained.h5` 생성
2. `keras2onnx.py` 실행 → `mnist_cnn_trained.onnx` 생성
3. `main.py` 실행 → 강건성 검증 결과 출력

## 4. 주요 실험 결과
- 10개 샘플 모두에서 UNSAT(안전) 판정, 즉 ±0.0007 섭동 범위 내에서 분류가 변하지 않음을 확인.
- 섭동 크기를 키우면 연산량이 급증하여 검증이 종료됨(실제 실험 한계).

## 5. 환경 및 재현
- Python 3.8, TensorFlow 2.x, tf2onnx, onnxruntime, maraboupy
- 코드와 모델, 환경설정 파일 포함. 재현을 위해 경로 및 패키지 버전을 맞춰주세요.

## 6. 참고
- Marabou: https://github.com/NeuralNetworkVerification/Marabou
- MNIST: https://keras.io/api/datasets/mnist/
