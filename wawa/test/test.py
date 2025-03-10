import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from wawa.models.resnet import get_model
from wawa.dataset.mnist import get_dataloaders
from wawa.utils.logger import get_logger

logger = get_logger(__name__)

def test_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    logger.info("테스트 시작...")

    with torch.no_grad():  # 테스트 시에는 그래디언트 계산 불필요
        for batch_idx, (images, labels) in enumerate(test_loader):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            # 10개 배치마다 로그 출력
            if (batch_idx + 1) % 10 == 0:
                batch_accuracy = 100 * correct / total
                logger.info(f"[Batch {batch_idx+1}/{len(test_loader)}] "
                            f"Loss: {loss.item():.4f}, "
                            f"Running Accuracy: {batch_accuracy:.2f}%")

    accuracy = 100 * correct / total
    logger.info(f"테스트 완료! Test Accuracy: {accuracy:.2f}%, Loss: {total_loss/len(test_loader):.4f}")

if __name__ == "__main__":
    config = {
        'batch_size': 64
    }
    _, test_loader = get_dataloaders(config['batch_size'])
    model = get_model()
    model.load_state_dict(torch.load("models/resnet_mnist.pth"))
    
    test_model(model, test_loader)
