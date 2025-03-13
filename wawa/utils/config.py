import yaml

def load_config(config_path="../../wawa/configs/config.yaml"):
    """YAML 설정 파일을 로드하는 함수"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
