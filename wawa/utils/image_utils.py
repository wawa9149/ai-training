import torchvision.transforms as transforms
from PIL import Image

def preprocess_image(image_path):
    """이미지를 MNIST 모델이 처리할 수 있도록 변환"""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # ResNet은 3채널 입력 필요
        transforms.Resize((224, 224)),  # ResNet의 기본 입력 크기 224x224
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # ImageNet 정규화 값
    ])

    image = Image.open(image_path)
    image = transform(image)
    return image.unsqueeze(0)  # 배치 차원 추가 (1, 3, 28, 28)
