import json
from pathlib import Path

CONFIG_FILE = Path("app_config.json")

def load_settings():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    return {}

def save_settings(settings):
    with open(CONFIG_FILE, "w") as f:
        json.dump(settings, f)
