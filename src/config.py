import os

CONFIG_NAME = "model"  # Default config name
config_name = os.getenv("CONFIG") or CONFIG_NAME

# Load corresponding config
from src.utils.utils import load_config

config = load_config(f"configs/{config_name}.py")
# Testing
if __name__ == "__main__":
    import rich

    from omegaconf import OmegaConf
    # 将配置对象转换为YAML格式的字符串，并使用rich.print打印出来，这样可以得到一个更加易读的输出。
    rich.print(OmegaConf.to_yaml(config))
