import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=32):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # ResNet50은 3채널 입력 필요
        transforms.Resize((28, 28)),  # 크기를 유지 (원래는 224x224가 기본)
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # MNIST 정규화
    ])

    # MNIST 훈련 데이터
    train_dataset = datasets.MNIST(root='data/mnist', train=True, transform=transform, download=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # MNIST 테스트 데이터
    test_dataset = datasets.MNIST(root='data/mnist', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader