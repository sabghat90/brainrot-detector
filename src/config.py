"""Loads settings.json and prestige_brainrots.json into plain dicts."""
import json
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(PROJECT_ROOT, "config")
SETTINGS_PATH = os.path.join(CONFIG_DIR, "settings.json")
PRESTIGE_PATH = os.path.join(CONFIG_DIR, "prestige_brainrots.json")


def load_settings(path=SETTINGS_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_prestige_list(path=PRESTIGE_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_settings(settings, path=SETTINGS_PATH):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
