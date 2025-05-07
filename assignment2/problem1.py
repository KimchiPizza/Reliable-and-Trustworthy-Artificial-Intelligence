# -*- coding: utf-8 -*-
"""
DeepXplore ResNet50 Differential Testing on CIFAR-10
Author: Kim Min-Yeong 
Date: 2025-05-01

해당 코드는 deepxplore을 import하지 않으나 deepxplore를 참고하여 작성됨
model1: resnet50_cifar10 외부 모델을 import함
model2: 직접 resnet50모델을 cifar10 데이터셋 학습시킨 모델을 사용.
이 코드는 이 모델을 학습을 시킬수도 있으나 현재는 학습 완료된 모델을 불러오도록 설정하였음. (저장된 모델이 없다면 새로 학습 시작함함)

"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from multiprocessing import freeze_support
import os
import urllib.request
from torch.nn.modules import dropout
import detectors
import timm
from tqdm import tqdm
import torch.optim as optim
import random
from collections import defaultdict
from neuron_coverage import NeuronCoverage

def init_coverage_tables(model1, model2):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    
    # Initialize coverage tables for each model
    for name, module in model1.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            for i in range(module.out_features if isinstance(module, nn.Linear) else module.out_channels):
                model_layer_dict1[(name, i)] = False
                
    for name, module in model2.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            for i in range(module.out_features if isinstance(module, nn.Linear) else module.out_channels):
                model_layer_dict2[(name, i)] = False
                
    return model_layer_dict1, model_layer_dict2

def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(list(model_layer_dict.keys()))
    return layer_name, index

def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    total_neurons = len(model_layer_dict)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)

def update_coverage(input_data, model, model_layer_dict, threshold=0):
    model.eval()
    hooks = []
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks for all layers
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    with torch.no_grad():
        _ = model(input_data)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Update coverage
    for name, activation in activations.items():
        if isinstance(activation, torch.Tensor):
            activation = activation.cpu().numpy()
            scaled = scale(activation)
            for i in range(scaled.shape[-1]):
                if np.mean(scaled[..., i]) > threshold and not model_layer_dict[(name, i)]:
                    model_layer_dict[(name, i)] = True

def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min() + 1e-8)
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled

def diverged(predictions1, predictions2, predictions3, target):
    if not predictions1 == predictions2:
        return True
    return False

def train_model(model, trainloader, criterion, optimizer, device, num_epochs=1):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        progress_bar = tqdm(trainloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            progress_bar.set_postfix({
                'loss': running_loss/(progress_bar.n+1),
                'acc': 100.*correct/total
            })
            
            # 메모리 해제
            del outputs, loss
            torch.cuda.empty_cache()
    
    return model

def save_model(model, path):
    """모델 저장"""
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")

def load_model(model, path, device):
    """모델 불러오기"""
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        print(f"Model loaded from {path}")
    return model

def main():
    # 랜덤 시드 고정 (재현성)
    seed = 420
    torch.manual_seed(seed)
    np.random.seed(seed)

    # GPU 사용 가능 여부 확인
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 모델 저장 경로 설정
    model2_path = 'model2_resnet50_cifar10.pth'

    # 1. CIFAR-10 데이터 불러오기
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=False, num_workers=0)

    # 2. 모델 초기화
    # Model 1: Pretrained ResNet50
    model1 = timm.create_model("resnet50_cifar10", pretrained=True)
    model1 = model1.to(device)
    model1.eval()
    
    # 모델1의 파라미터 고정
    for param in model1.parameters():
        param.requires_grad = False

    # Model 2: ResNet50 (저장된 모델 불러오기)
    model2 = timm.create_model("resnet50_cifar10", pretrained=False)
    model2 = model2.to(device)
    
    # 저장된 모델 불러오기
    if os.path.exists(model2_path):
        model2 = load_model(model2, model2_path, device)
        print("Loaded existing model for continued training")
    else:
        print("No existing model found, starting from scratch")
    
    # Model 2 추가 학습
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model2.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    
    print("Training Model 2...")
    '''
    현재 num_epochs = 0 으로 학습을 건너뜀.
    '''
    model2 = train_model(model2, trainloader, criterion, optimizer, device, num_epochs=0)
    model2.eval()
    
    # 학습된 모델 저장
    save_model(model2, model2_path)

    # 3. NeuronCoverage 객체 생성
    nc1 = NeuronCoverage(model1)
    nc2 = NeuronCoverage(model2)

    # 4. 테스트 실행
    suspicious_cases = []
    num_samples = 100
    processed_samples = 0
    
    # 정확도 계산을 위한 변수
    correct1 = 0
    correct2 = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            if processed_samples >= num_samples:
                break
                
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs1 = model1(inputs)
            outputs2 = model2(inputs)
            preds1 = torch.argmax(outputs1, dim=1)
            preds2 = torch.argmax(outputs2, dim=1)
            
            total += labels.size(0)
            correct1 += (preds1 == labels).sum().item()
            correct2 += (preds2 == labels).sum().item()
            
            for i in range(len(labels)):
                if processed_samples >= num_samples:
                    break
                    
                if preds1[i] != preds2[i]:
                    suspicious_cases.append({
                        'index': processed_samples,
                        'true_label': labels[i].item(),
                        'model1_pred': preds1[i].item(),
                        'model2_pred': preds2[i].item()
                    })
                    print(f"[!] Suspicious case at idx {processed_samples}: GT={labels[i].item()} | M1={preds1[i].item()} | M2={preds2[i].item()}")

                # 커버리지 업데이트
                input_np = inputs[i:i+1].cpu().numpy()
                nc1.update_coverage(input_np)
                nc2.update_coverage(input_np)
                
                processed_samples += 1
            
            # 메모리 해제
            del outputs1, outputs2, preds1, preds2
            torch.cuda.empty_cache()

    # 결과 출력
    print("\nModel 1 (Pretrained) Test Accuracy: {:.2f}%".format(100 * correct1 / total))
    print("Model 2 (Trained) Test Accuracy: {:.2f}%".format(100 * correct2 / total))
    print("\nModel 1 Neuron Coverage: {:.2f}%".format(nc1.coverage() * 100))
    print("Model 2 Neuron Coverage: {:.2f}%".format(nc2.coverage() * 100))

    # suspicious_cases.json 결과 저장
    import json
    with open('suspicious_cases.json', 'w') as f:
        json.dump(suspicious_cases, f, indent=2)

if __name__ == '__main__':
    freeze_support()
    main()

