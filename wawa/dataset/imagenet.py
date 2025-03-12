import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=32):
    # imageNet 데이터셋을 사용하기 위한 전처리 과정
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # ResNet은 3채널 입력 필요
        transforms.Resize((224, 224)),  # ResNet의 기본 입력 크기 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet 정규화 값
    ])

    # imageNet 훈련 데이터
    train_dataset = datasets.ImageNet(root='data/imagenet', split='train', transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # imageNet 테스트 데이터
    test_dataset = datasets.ImageNet(root='data/imagenet', split='val', transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
