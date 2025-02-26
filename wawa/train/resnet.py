import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50

# ✅ 1. 하이퍼파라미터 설정
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 2. 데이터셋 로드 (MNIST → 1채널 흑백 이미지)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # ResNet은 3채널 입력을 기대하므로 변환
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.MNIST(root="data/mnist", train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ✅ 3. ResNet50 모델 정의 (입력 채널 수정)
class CustomResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomResNet, self).__init__()
        self.model = resnet50(pretrained=True)  # 사전 학습된 모델 사용
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 입력 1→3채널
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)  # 출력 노드 10개 (MNIST)

    def forward(self, x):
        return self.model(x)

model = CustomResNet().to(DEVICE)

# ✅ 4. 손실 함수 및 옵티마이저 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ✅ 5. 학습 루프
def train():
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

    # ✅ 학습된 모델 저장
    torch.save(model.state_dict(), "models/resnet_mnist.pth")
    print("모델 저장 완료: models/resnet_mnist.pth")