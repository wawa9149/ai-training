import torch
import os
import matplotlib.pyplot as plt
from wawa.models.resnet import get_model
from wawa.dataset.mnist import get_dataloaders
from wawa.utils.logger import get_logger

logger = get_logger(__name__)

def infer(model, image, label=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    
    if label is not None:
        logger.info(f"실제 값: {label}, 예측 값: {predicted.item()}")
    else:
        logger.info(f"예측 값: {predicted.item()}")

    # 이미지 시각화 (채널 차원 제거 후)
    image = image.cpu().squeeze()  # 차원 제거
    if image.ndimension() == 3:  # 이미지가 3채널로 되어 있을 경우 1채널로 변경
        image = image[0]  # 첫 번째 채널만 사용
    
    # 이미지 저장
    save_dir = "data/test"  # 저장할 폴더 지정
    os.makedirs(save_dir, exist_ok=True)  # 폴더 없으면 생성

    image_filename = os.path.join(save_dir, f"predicted_{predicted.item()}.png") # 파일명에 예측된 값 포함
    plt.imshow(image, cmap="gray")  # 채널을 1개로 바꾸고 이미지 시각화
    plt.title(f"Predicted: {predicted.item()}")
    plt.savefig(image_filename)  # 이미지 파일로 저장
    logger.info(f"이미지를 {image_filename}로 저장했습니다.")  # 저장된 파일 경로 로그로 출력
    plt.close()  # plt.show() 대신 사용하여 창을 띄우지 않고 바로 파일로 저장