from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    config = None
    with open(file=config_path, mode="r", encoding="utf-8") as f:
        config = yaml.safe_load(stream=f)
    return config
