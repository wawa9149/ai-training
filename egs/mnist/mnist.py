#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST Training Script

This script is used to train a neural network on the MNIST dataset.
It includes data loading, model definition, training, and evaluation.

Author: Nayoung Park
Date: February 26, 2025
"""

from wawa.train.resnet import train
from parser import parse_args
from wawa.utils.logger import get_logger 
logger = get_logger(__name__)

class MNIST:
    def __init__(self, args):
        self.args = args
        logger.info(f"Arguments: {args}")

    def train(self):
        logger.info("Training the model")
        train()
        logger.info("Model training complete")

    def test(self):
        logger.info("Testing the model")

    def infer(self):
        logger.info("Inferring with the model")

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
    mnist = MNIST(args)
    mnist.run()


if __name__ == "__main__":
    main()