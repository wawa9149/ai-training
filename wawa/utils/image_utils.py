import torchvision.transforms as transforms
from PIL import Image

def preprocess_image(image_path):
    """이미지를 MNIST 모델이 처리할 수 있도록 변환"""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # ResNet50은 3채널 입력 필요
        transforms.Resize((28, 28)),  # 크기 조정
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # MNIST 정규화
    ])

    image = Image.open(image_path).convert("L")  # 흑백 변환
    image = transform(image)
    return image.unsqueeze(0)  # 배치 차원 추가 (1, 3, 28, 28)
