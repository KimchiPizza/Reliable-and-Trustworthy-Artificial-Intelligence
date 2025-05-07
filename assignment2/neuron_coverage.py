import torch
import numpy as np
from collections import defaultdict

class NeuronCoverage:
    def __init__(self, model, threshold=0.5):
        self.model = model
        self.threshold = threshold
        self.coverage_dict = defaultdict(bool)
        self.device = next(model.parameters()).device  # 모델의 디바이스 가져오기
        self._init_coverage_dict()
        
    def _init_coverage_dict(self):
        """모델의 모든 뉴런에 대한 커버리지 딕셔너리 초기화"""
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                for i in range(module.out_features if isinstance(module, torch.nn.Linear) else module.out_channels):
                    self.coverage_dict[(name, i)] = False
    
    def update_coverage(self, input_data):
        """입력 데이터에 대한 뉴런 커버리지 업데이트"""
        self.model.eval()
        hooks = []
        activations = {}
        
        def hook_fn(name):
            def hook(module, input, output):
                activations[name] = output.detach()
            return hook
        
        # 모든 레이어에 대한 훅 등록
        for name, module in self.model.named_modules():
            if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # 순전파 (입력 데이터를 모델과 같은 디바이스로 이동)
        with torch.no_grad():
            input_tensor = torch.from_numpy(input_data).float().to(self.device)
            _ = self.model(input_tensor)
        
        # 훅 제거
        for hook in hooks:
            hook.remove()
        
        # 커버리지 업데이트
        for name, activation in activations.items():
            if isinstance(activation, torch.Tensor):
                activation = activation.cpu().numpy()
                scaled = self._scale(activation)
                for i in range(scaled.shape[-1]):
                    if np.mean(scaled[..., i]) > self.threshold and not self.coverage_dict[(name, i)]:
                        self.coverage_dict[(name, i)] = True
    
    def _scale(self, intermediate_layer_output, rmax=1, rmin=0):
        """활성화값을 0-1 범위로 스케일링"""
        X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
            intermediate_layer_output.max() - intermediate_layer_output.min() + 1e-8)
        X_scaled = X_std * (rmax - rmin) + rmin
        return X_scaled
    
    def coverage(self):
        """현재 커버리지 비율 반환"""
        covered_neurons = len([v for v in self.coverage_dict.values() if v])
        total_neurons = len(self.coverage_dict)
        return covered_neurons / float(total_neurons) if total_neurons > 0 else 0
    
    def get_coverage_dict(self):
        """현재 커버리지 딕셔너리 반환"""
        return dict(self.coverage_dict) 