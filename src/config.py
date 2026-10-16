"""
Loads config.yaml. Every other script gets its settings from here, so that
no path, date, threshold or seed is ever written twice.

Usage:
    from src.config import load_config, set_seeds
    cfg = load_config()
    print(cfg["period"]["start"])
"""

import os
import random

import yaml

# The project root is the folder that contains config.yaml, i.e. one level
# above src/. Scripts are always run from the root, but resolving it this way
# means they work even if they are not.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(ROOT, "config.yaml")


def load_config(path=CONFIG_PATH):
    """Read config.yaml into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(*parts):
    """Turn a path from config.yaml into an absolute path."""
    return os.path.join(ROOT, *parts)


def set_seeds(seed):
    """Fix the random seeds we can fix without importing the heavy libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
