"""Plotting helpers for diagnostics."""
import matplotlib.pyplot as plt
import json, pathlib

def plot_curves(log_file: str, out_png: str):
    data = json.load(open(log_file))
    steps = range(len(data['train_loss']))
    plt.figure()
    plt.plot(steps, data['train_loss'], label='train_loss')
    plt.plot(steps, data['val_acc'], label='val_acc')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(out_png, bbox_inches='tight')
