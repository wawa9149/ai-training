import torch.nn as nn
from torchvision.models import resnet50

def get_model():
    model = resnet50(weights=True) # 사전 학습된 모델 사용
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 입력 크기 맞춤
    model.fc = nn.Linear(2048, 10)  # MNIST는 10개의 클래스
    return model
