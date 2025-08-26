from loguru import logger
import yaml
import os
from typing import Dict


class Config:
    def __init__(self):
        if os.environ.get("MODE", "dev") == "prod":
            self.conf_filename = "./config/conf-prod.yml"
        else:
            self.conf_filename = "./config/conf-dev.yml"
        self._config = {}
        self.reload()

    def __load_file(self, filepath: str):
        try:
            with open(filepath, "r") as fr:
                return fr.read()
        except Exception:
            logger.opt(exception=False, colors=True).warning(
                f"Loading {filepath} failed"
            )
            return None

    def __getitem__(self, key):
        return self._config.get(key, None)

    def __getattr__(self, name):
        return self._config.get(name, None)

    def get(self, key: str, default_value=None):
        return self._config.get(key, default_value)

    def reload(self):
        logger.info("Config loaded")
        with open(self.conf_filename, "r") as fr:
            self._config: Dict = yaml.safe_load(fr)


config = Config()
