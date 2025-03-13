#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImageNet Training Script

This script is used to train a neural network on the ImageNet dataset.
It includes data loading, model definition, training, and evaluation.

Author: Nayoung Park
Date: February 26, 2025
"""
import torch
from wawa.dataset.imagenet import get_dataloaders
from wawa.models.resnet import get_model
from wawa.train.train import train
from wawa.test.test import test_model as test
from wawa.infer.infer import infer
from wawa.utils.image_utils import preprocess_image
import os
from parser import parse_args
from wawa.utils.logger import get_logger 
from wawa.utils.config import load_config

logger = get_logger(__name__)
MODEL_PATH = "models/resnet_imagenet.pth"

class ImageNet:
    def __init__(self, args):
        self.args = args
        logger.info(f"Arguments: {args}")

    def train(self):
        logger.info("Training the model")
        config = load_config()
        train_loader, _ = get_dataloaders(config['batch_size'])
        model = get_model(num_classes=10)
        train(model, 'resnet_imagenet', train_loader, config)

    def test(self):
        logger.info("Testing the model")
        config = load_config()
        _, test_loader = get_dataloaders(config['batch_size'])
        model = get_model(num_classes=10)
        
        try:
            model.load_state_dict(torch.load(MODEL_PATH))
            logger.info(f"Loaded model weights from {MODEL_PATH}")
        except FileNotFoundError:
            logger.error(f"Model file not found: {MODEL_PATH}")
            return
        
        test(model, test_loader)

    def infer(self):
        logger.info("Running inference")
        model = get_model(num_classes=10)
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file not found: {MODEL_PATH}")
            return
        
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        logger.info(f"Loaded model weights from {MODEL_PATH}")

        # ✅ 직접 찍은 이미지로 추론
        if self.args.image_path:
            if not os.path.exists(self.args.image_path):
                logger.error(f"Image file not found: {self.args.image_path}")
                return
            
            image = preprocess_image(self.args.image_path)  # 이미지 전처리
            logger.info(f"Inferring with image: {self.args.image_path}")
            infer(model, image)

        else:
            # ✅ 기본적으로 ImageNet 데이터셋 사용
            config = load_config()
            _, test_loader = get_dataloaders(config['batch_size'])
            infer(model, test_loader)

    def run(self):
        if self.args.task == "train":
            self.train()
        elif self.args.task == "test":
            self.test()
        elif self.args.task == "infer":
            self.infer()
        else:
            raise ValueError(f"Invalid task: {self.args.task}")

def main():
    args = parse_args()
    logger.info(f"Arguments: {args}")
    imagenet = ImageNet(args)
    imagenet.run()

if __name__ == "__main__":
    main()
