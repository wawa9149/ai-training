import torch
import torch.optim as optim
import torch.nn as nn
from wawa.utils.logger import get_logger

logger = get_logger(__name__)

def train(model, train_loader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    logger.info(f"🔥 Training started | Device: {device} | Epochs: {config['epochs']} | Batch size: {config['batch_size']}")

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        logger.info(f"📢 Epoch {epoch+1}/{config['epochs']} started...")

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 정확도 계산
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 일정 간격마다 로그 출력 (ex. 10개 배치마다)
            if (batch_idx + 1) % 200 == 0:
                logger.info(f"  🏃 Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct / total

        logger.info(f"✅ Epoch {epoch+1} finished | Avg Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")

    torch.save(model.state_dict(), 'models/resnet_mnist.pth')
    logger.info(f"💾 Model saved: models/resnet_mnist.pth | Final Loss: {epoch_loss:.4f} | Final Accuracy: {epoch_accuracy:.2f}%")
