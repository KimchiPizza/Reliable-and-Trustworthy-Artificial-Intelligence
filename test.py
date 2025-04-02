import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

torch.manual_seed(12000)  # 재현성 보장
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False    

def fgsm_targeted(model, x, target, eps):
    """
    Targeted FGSM 공격 구현
    
    Args:
        model: 학습된 모델
        x: 입력 이미지 텐서 (B, C, H, W)
        target: 목표 클래스 인덱스 (B,)
        eps: 공격 강도 (0~1)
    
    Returns:
        adversarial example: 적대적 예제 이미지
    """
    # 입력 텐서를 복사하고 그래디언트 계산 활성화
    x_adv = x.clone().detach().requires_grad_(True)
    
    # 모델 추론 및 손실 계산
    output = model(x_adv)
    loss = F.cross_entropy(output, target)  # 목표 클래스에 대한 손실 계산
    
    # 그래디언트 계산
    grad = torch.autograd.grad(loss, x_adv)[0]
    
    # perturbation 생성 및 적용
    perturbation = eps * grad.sign()  # 그래디언트 방향으로 perturbation 생성
    x_adv = x_adv - perturbation  # 타겟 공격은 목표 클래스 방향으로 이동
    
    # 이미지 도메인으로 클램핑 ([0, 1] 범위 유지)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    
    return x_adv


def fgsm_untargeted(model, x, true_labels, eps):
    """
    Untargeted FGSM 공격 구현
    
    Args:
        model: 학습된 모델
        x: 입력 이미지 텐서 (B, C, H, W)
        true_labels: 실제 정답 레이블 (B,)
        eps: 공격 강도 (0~1)
    
    Returns:
        adversarial example: 적대적 예제 이미지
    """
    x_adv = x.clone().detach().requires_grad_(True)
    
    # 실제 레이블에 대한 손실 계산 (최대화)
    output = model(x_adv)
    loss = F.cross_entropy(output, true_labels)
    
    # 그래디언트 계산 및 perturbation 생성
    grad = torch.autograd.grad(loss, x_adv)[0]
    perturbation = eps * grad.sign()
    
    # perturbation 추가 (원본 이미지 손실 증가 방향)
    x_adv = x_adv + perturbation
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach()
    
    return x_adv

def pgd_targeted(model, x, target, k, eps, eps_step):
    """
    Targeted PGD 공격 구현
    
    Args:
        model: 학습된 모델
        x: 입력 이미지 텐서 (B, C, H, W)
        target: 목표 클래스 인덱스 (B,)
        k: 반복 횟수
        eps: 최대 perturbation 크기 (0~1)
        eps_step: 단계별 perturbation 크기
        
    Returns:
        adversarial example: 적대적 예제 이미지
    """
    # 초기 perturbation 생성 (랜덤 초기화)
    x_orig = x.clone().detach()
    x_adv = x_orig + torch.zeros_like(x).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0).detach().requires_grad_(True)
    
    for _ in range(k):
        # 목표 클래스 손실 계산
        output = model(x_adv)
        loss = F.cross_entropy(output, target)
        
        # 그래디언트 계산
        grad = torch.autograd.grad(loss, x_adv)[0]
        
        # perturbation 업데이트 (목표 클래스 방향)
        x_adv = x_adv - eps_step * grad.sign()  # 타겟 공격 방향
        
        # 프로젝션 단계 (eps-ball 내부로 클램핑)
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        
        # 이미지 도메인으로 클램핑
        x_adv = torch.clamp(x_adv, 0.0, 1.0).detach().requires_grad_(True)
        
    return x_adv


# 데이터셋 로드 함수
def load_dataset(dataset_name):
    """
    데이터셋 로드 함수
    
    Args:
        dataset_name: 사용할 데이터셋 이름 ("MNIST" 또는 "CIFAR-10")
    
    Returns:
        train_set: 학습 데이터셋
        test_set: 테스트 데이터셋
        num_classes: 클래스 개수
        img_size: 이미지 크기 정보 (채널 수 포함)
    """
    if dataset_name == "CIFAR-10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, transform=transform)
        num_classes = 10
        img_size = (3, 32, 32)
    elif dataset_name == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, transform=transform)
        num_classes = 10
        img_size = (1, 28, 28)
    else:
        raise ValueError("Unsupported dataset. Choose either 'MNIST' or 'CIFAR-10'.")
    
    return train_set, test_set, num_classes, img_size
# 모델 정의 함수 (MNIST/CIFAR-10 처리용)
def get_model(dataset_name, num_classes):
    """
    데이터셋 이름에 따라 적절한 모델 구조를 반환하는 함수
    
    Args:
        dataset_name (str): 사용할 데이터셋 이름 ("MNIST" 또는 "CIFAR-10")
        num_classes (int): 분류할 클래스 개수
    
    Returns:
        nn.Module: 초기화된 모델 객체
    
    주요 기능:
        - CIFAR-10: TorchVision의 사전 학습된 ResNet-18 모델 활용 (전이 학습)
        - MNIST: 커스텀 CNN 아키텍처 구현
    """
    if dataset_name == "CIFAR-10":
        # CIFAR-10: 사전 학습된 ResNet-18 모델 사용 (출처: TorchVision)
        # 논문 인용: He et al. "Deep Residual Learning for Image Recognition", CVPR 2016
        model = torchvision.models.resnet18(pretrained=True)
        # 마지막 완전 연결층을 데이터셋에 맞게 수정
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    elif dataset_name == "MNIST":
        # MNIST: 단순 CNN 아키텍처 직접 구현
        class MNIST_CNN(nn.Module):
            def __init__(self):
                super(MNIST_CNN, self).__init__()
                # 특징 추출기 (Feature Extractor)
                self.conv_layers = nn.Sequential(
                    nn.Conv2d(1, 32, 3, padding=1),  # 입력 채널 1, 출력 채널 32
                    nn.ReLU(inplace=True),            # 비선형 활성화 함수
                    nn.MaxPool2d(2),                  # 공간 차원 절반으로 축소
                    nn.Conv2d(32, 64, 3, padding=1),  # 채널 수 64로 증가
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2)                   # 최종 특징맵 크기: 7x7
                )
                # 분류기 (Classifier)
                self.fc_layers = nn.Sequential(
                    nn.Linear(7*7*64, 256),           # 7x7x64 -> 256
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),                  # 과적합 방지를 위한 드롭아웃
                    nn.Linear(256, num_classes)       # 최종 분류층
                )

            def forward(self, x):
                # 순전파 과정 정의
                x = self.conv_layers(x)               # 특징 추출
                x = x.view(x.size(0), -1)             # 평탄화(Flatten)
                return self.fc_layers(x)              # 분류 수행

        model = MNIST_CNN()
    else:
        raise ValueError("지원하지 않는 데이터셋입니다. 'MNIST' 또는 'CIFAR-10'을 선택하세요.")
    
    return model


def train_model(model, train_loader, EPOCHS, device):
    """
    모델 학습을 수행하는 함수
    
    Args:
        model (nn.Module): 학습할 모델 객체
        train_loader (DataLoader): 학습 데이터 로더
        EPOCHS (int): 전체 학습 에포크 수
        device (str): 학습에 사용할 장치 ('cuda' 또는 'cpu')
    
    주요 기능:
        - Adam 옵티마이저 사용
        - 교차 엔트로피 손실 함수 활용
        - 에포크별 검증 정확도 출력
    """
    optimizer = torch.optim.Adam(model.parameters())  # Adam 옵티마이저
    criterion = nn.CrossEntropyLoss()                 # 손실 함수

    for epoch in range(EPOCHS):
        model.train()  # 학습 모드
        for batch_idx, (data, target) in enumerate(train_loader):
            # 데이터 장치 이동
            data, target = data.to(device), target.to(device)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순전파 및 손실 계산
            output = model(data)
            loss = criterion(output, target)
            
            # 역전파 및 가중치 업데이트
            loss.backward()
            optimizer.step()

        # 에포크별 검증
        model.eval()  # 평가 모드
        correct = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1)  # 예측 클래스
                correct += pred.eq(target).sum().item()  # 정확한 예측 수 집계

        # 정확도 계산 및 출력
        acc = 100. * correct / len(train_loader.dataset)
        print(f'Epoch {epoch+1}: 검증 정확도 {acc:.2f}%')


def plot_accuracy_vs_epsilon(epsilons, success_rates):
    """
    ε 값에 따른 공격 성공률 변화를 시각화하는 함수
    
    Args:
        epsilons (list): 테스트한 epsilon 값 리스트
        success_rates (list): 각 epsilon에 해당하는 공격 성공률
    
    시각화 요소:
        - 파란색 원 마커와 실선으로 성공률 표시
        - 그리드 라인 추가
        - 축 레이블 및 제목 설정
    """
    plt.figure(figsize=(10, 6))
    plt.plot(epsilons, success_rates, 'bo-', label='공격 성공률')
    plt.title("Performance Comparison of Adversarial Attack Methods", fontsize=16)
    plt.xlabel("Epsilon (ε)", fontsize=14)
    plt.ylabel("Attack Success Rate (%)", fontsize=14)
    plt.xticks(epsilons)
    plt.yticks(torch.linspace(0, 1, 11))  # 0%~100% 범위를 10% 단위로 표시
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()


def plot_all_results():
    """
    모든 실험 결과를 종합 비교하는 시각화 함수
    
    주요 기능:
        - FGSM Targeted/Untargeted 및 PGD Targeted 결과 비교
        - MNIST(파랑)와 CIFAR-10(빨강) 결과 병렬 표시
        - 마커 모양으로 공격 방법 구분
        - 범례를 우측 상단에 배치
    
    요구 사항:
        - result_history 딕셔너리에 데이터 저장 필요
        - 형식: {'MNIST': {'FGSM_Targeted': [...], ...}, 'CIFAR-10': {...}}
    """
    plt.figure(figsize=(16, 8))
    
    # 시각화 스타일 설정
    markers = {'FGSM_Targeted': 'o', 'FGSM_Untargeted': 's', 'PGD_Targeted': 'D'}
    colors = {'MNIST': 'blue', 'CIFAR-10': 'red'}
    
    # 공격 방법별 순회
    for attack in ['FGSM_Targeted', 'FGSM_Untargeted', 'PGD_Targeted']:
        # 데이터셋별 플롯
        for dataset in ['MNIST', 'CIFAR-10']:
            epsilons = [0.1, 0.3, 0.5, 0.7, 0.9]
            rates = result_history[dataset][attack]
            
            # 선 스타일 결정
            linestyle = '--' if 'Untargeted' in attack else '-'
            
            plt.plot(epsilons, rates, 
                     marker=markers[attack], 
                     color=colors[dataset],
                     linestyle=linestyle,
                     linewidth=2,
                     markersize=10,
                     label=f'{dataset} {attack}')

    plt.title("Performance Comparison of Adversarial Attack Methods", fontsize=16)
    plt.xlabel("Epsilon (ε)", fontsize=14)
    plt.ylabel("Attack Success Rate (%)", fontsize=14)
    plt.xticks(epsilons)
    plt.yticks([i/10 for i in range(0, 11)])  # 0.0 ~ 1.0 범위
    plt.ylim(0, 1.0)  # y축 범위 고정
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 범례 외부 배치
    plt.tight_layout()
    plt.show()




def main(dataset_name):
    """메인 실행 함수: 데이터셋별 모델 학습 및 적대적 공격 평가 수행"""
    
    # 1. 디바이스 설정 (GPU 우선 사용)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 2. 데이터셋 및 모델 로드
    train_set, test_set, num_classes, _ = load_dataset(dataset_name)
    model = get_model(dataset_name, num_classes).to(device)  # 모델을 선택한 디바이스로 이동

    # 3. 재현성 보장을 위한 고정된 데이터 로더 생성
    fixed_generator = torch.Generator().manual_seed(12000)  # 동일한 셔플링 결과 보장
    train_loader = DataLoader(train_set, 
                            batch_size=64, 
                            shuffle=True,
                            generator=fixed_generator)

    # 4. 모델 학습
    print(f"\nTraining {dataset_name} model...")
    train_model(model=model, EPOCHS=3, train_loader=train_loader, device=device)

    # 5. Prepare test samples
    sample_size = 500  # For statistical reliability
    indices = torch.randint(0, len(test_set), (sample_size,))  # Generate random indices
    test_sample = torch.stack([test_set[i][0].to(device) for i in indices])  # Move to GPU
    true_labels = torch.tensor([test_set[i][1] for i in indices]).to(device)  # True labels

    # 6. Calculate original predictions
    with torch.no_grad():
        orig_output = model(test_sample)
        orig_pred = orig_output.argmax(dim=1)

    # 7. Generate target labels (force different from original prediction)
    target = []
    for pred in orig_pred:
        while True:  # Repeat until different class is selected
            t = torch.randint(0, 10, (1,)).item()
            if t != pred.item():
                target.append(t)
                break
    target = torch.tensor(target).to(device)

    # 8. FGSM Targeted Attack Evaluation
    print("\n[FGSM Targeted Attack Evaluation]")
    epsilons = [0.1, 0.3, 0.5, 0.7, 0.9]  # List of epsilon values to test
    success_rates = []
    
    for eps in epsilons:
        # 8-1. Perform FGSM attack
        adv_images = fgsm_targeted(model, test_sample, target, eps)
        
        # 8-2. Analyze attack results
        with torch.no_grad():
            adv_output = model(adv_images)
            adv_pred = adv_output.argmax(dim=1)
        
        # 8-3. Calculate success rate (ratio matching target class)
        success_rate = (adv_pred == target).float().mean().item()
        success_rates.append(success_rate)
        print(f"ε={eps:.1f} | Attack Success Rate: {success_rate*100:.2f}%")
    
    # 8-4. Save results
    result_history[dataset_name]['FGSM_Targeted'] = success_rates

    # 9. FGSM Untargeted Attack Evaluation
    print("\n[FGSM Untargeted Attack Evaluation]")
    success_rates = []
    for eps in epsilons:
        # 9-1. Perform FGSM attack (using true labels)
        adv_images = fgsm_untargeted(model, test_sample, true_labels, eps)
        
        # 9-2. Analyze results (success if prediction differs from true label)
        with torch.no_grad():
            adv_output = model(adv_images)
            adv_pred = adv_output.argmax(dim=1)
        
        success_rate = (adv_pred != true_labels).float().mean().item()
        success_rates.append(success_rate)
        print(f"ε={eps:.1f} | Attack Success Rate: {success_rate*100:.2f}%")
        
    result_history[dataset_name]['FGSM_Untargeted'] = success_rates

    # 10. PGD Attack Evaluation
    print("\n[PGD Targeted Attack Evaluation]")
    success_rates = []
    for eps in epsilons:
        # 10-1. Set PGD attack parameters
        adv_images = pgd_targeted(
            model=model,
            x=test_sample,
            target=target,
            k=40,         # Number of iterations (standard value from paper)
            eps=eps,      # Maximum perturbation size
            eps_step=0.3  # Step size for updates
        )
        
        # 10-2. Analyze results
        with torch.no_grad():
            adv_output = model(adv_images)
            adv_pred = adv_output.argmax(dim=1)
        
        success_rate = (adv_pred == target).float().mean().item()
        success_rates.append(success_rate)
        print(f"ε={eps:.1f} | Attack Success Rate: {success_rate*100:.2f}%")
        
    result_history[dataset_name]['PGD_Targeted'] = success_rates

# 11. Global result storage dictionary
result_history = {
    'MNIST': {'FGSM_Targeted': [], 'FGSM_Untargeted': [], 'PGD_Targeted': []},
    'CIFAR-10': {'FGSM_Targeted': [], 'FGSM_Untargeted': [], 'PGD_Targeted': []}
}

# 12. Main execution block
if __name__ == "__main__":
    # Sequential evaluation for MNIST/CIFAR-10
    main("MNIST")        # Evaluate on MNIST dataset
    main("CIFAR-10")     # Evaluate on CIFAR-10 dataset
    
    # 13. Visualize comprehensive results
    plot_all_results()   # Compare all results in a single plot
