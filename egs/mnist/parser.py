#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MNIST Parser
"""
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="MNIST Parser")
    parser.add_argument("--task", type=str, default="train", help="Task to perform: train, test, or infer")
    parser.add_argument("--model", type=str, default="resnet", help="Model to use: resnet, mlp")
    parser.add_argument("--image_path", type=str, default=None, help="Path to the image for inference (only used in infer mode)")

    return parser.parse_args()