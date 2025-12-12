# src/persist.py
"""
Minimal persistence helpers for saving models and results.
"""
import joblib
from pathlib import Path

def save_model(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)

def load_model(path):
    return joblib.load(path)
