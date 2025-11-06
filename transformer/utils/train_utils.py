import math
import matplotlib.pyplot as plt
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_ppl(loss):
    return math.exp(loss) if loss < 20 else float("inf")

def plot_training_curves(results, out_dir):
    plt.figure(figsize=(10,6))
    for r in results:
        plt.plot(r['train'], label=f"{r['name']}_train_loss")
        plt.plot(r['val'], '--', label=f"{r['name']}_val_loss")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/encoder_loss_curves.png")

    plt.figure(figsize=(10,6))
    for r in results:
        plt.plot(r['train_ppl'], label=f"{r['name']}_train_ppl")
        plt.plot(r['val_ppl'], '--', label=f"{r['name']}_val_ppl")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("PPL")
    plt.title("Training / Validation Perplexity")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/encoder_ppl_curves.png")




