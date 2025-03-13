import yaml
from dataset.mnist import get_dataloaders
from models.resnet import get_model
from train.train import train_model
from wawa.test.test import evaluate_model

def load_config(config_path="configs/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    train_loader, test_loader = get_dataloaders(config['batch_size'])
    model = get_model()
    train_model(model, train_loader, config)
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
