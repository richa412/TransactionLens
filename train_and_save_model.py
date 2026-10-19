"""
train_and_save_model.py
-----------------------
Convenience wrapper — run this from the project root to train and persist
the Isolation Forest model.

    python train_and_save_model.py

This is equivalent to running:
    python src/model_training.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from model_training import run_training

if __name__ == "__main__":
    run_training()
