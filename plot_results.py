import json
import os

import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(PROJECT_DIR, "results.json")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_SECONDARY,
    "xtick.color": TEXT_SECONDARY, "ytick.color": TEXT_SECONDARY,
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
})


def style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, loc="left", color=TEXT_PRIMARY, fontweight="bold", fontsize=13, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_accuracy_comparison(series, title, filename):
    fig, ax = plt.subplots(figsize=(8, 4.8))
    n_epochs = len(series[0][1]) - 1
    epochs = np.arange(1, n_epochs + 1)

    end_labels = []
    for (label, accs), color in zip(series, SERIES):
        ax.plot(epochs, accs[1:], color=color, linewidth=2, marker="o", markersize=5, label=label)
        end_labels.append([accs[-1], label, color])

    end_labels.sort()
    y_positions = [v for v, _, _ in end_labels]
    for i in range(1, len(y_positions)):
        y_positions[i] = max(y_positions[i], y_positions[i - 1] + 0.32)
    for (value, label, color), y in zip(end_labels, y_positions):
        ax.annotate(f"{label}  {value:.2f}%", (n_epochs, value), xytext=(n_epochs + 0.25, y),
                    color=TEXT_PRIMARY, fontsize=10, va="center",
                    arrowprops=dict(arrowstyle="-", color=color, lw=1) if abs(y - value) > 0.05 else None)

    ax.set_xlim(0.6, n_epochs + 2.9)
    ax.set_xticks(epochs)
    style_axis(ax, title, "Epoch", "Validation accuracy (%)")
    ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(0.8, 0.02), fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, filename), dpi=160)
    plt.close(fig)


def plot_long_training(history, title, filename):
    n_epochs = len(history["val_accuracies"]) - 1
    epochs = np.arange(n_epochs + 1)
    train_losses, val_losses, val_accs = history["train_losses"], history["val_losses"], history["val_accuracies"]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax_loss.plot(epochs[1:], train_losses[1:], color=SERIES[0], linewidth=2, label="Train loss")
    ax_loss.plot(epochs[1:], val_losses[1:], color=SERIES[1], linewidth=2, label="Validation loss")
    best = int(np.argmin(val_losses))
    ax_loss.plot(best, val_losses[best], "o", color=SERIES[1], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    ax_loss.annotate(f"lowest val loss\nepoch {best}: {val_losses[best]:.4f}", (best, val_losses[best]),
                     xytext=(max(best - 9, 1), val_losses[best] * 1.6), color=TEXT_PRIMARY, fontsize=9.5,
                     arrowprops=dict(arrowstyle="-", color=TEXT_SECONDARY, lw=0.8))
    ax_loss.text(n_epochs + 0.4, train_losses[-1], "Train", color=TEXT_PRIMARY, va="center", fontsize=10)
    ax_loss.text(n_epochs + 0.4, val_losses[-1], "Validation", color=TEXT_PRIMARY, va="center", fontsize=10)
    ax_loss.set_yscale("log")
    ax_loss.set_xlim(0, n_epochs + 4.5)
    style_axis(ax_loss, "Loss (log scale)", "Epoch", "Cross-entropy")
    ax_loss.legend(frameon=False, fontsize=10)

    ax_acc.plot(epochs[1:], val_accs[1:], color=SERIES[2], linewidth=2)
    best = int(np.argmax(val_accs))
    ax_acc.plot(best, val_accs[best], "o", color=SERIES[2], markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    ax_acc.annotate(f"best {val_accs[best]:.2f}% (epoch {best})", (best, val_accs[best]),
                    xytext=(n_epochs * 0.45, val_accs[best] - 1.5), color=TEXT_PRIMARY, fontsize=9.5,
                    arrowprops=dict(arrowstyle="-", color=TEXT_SECONDARY, lw=0.8))
    style_axis(ax_acc, "Validation accuracy", "Epoch", "Accuracy (%)")

    fig.suptitle(title, x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, filename), dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    with open(RESULTS_PATH) as f:
        results = json.load(f)

    baseline_accs = results["epochs=30"]["val_accuracies"][:11]

    plot_accuracy_comparison(
        [("lr = 0.01", results["lr=0.01"]["val_accuracies"]),
         ("lr = 0.05", results["lr=0.05"]["val_accuracies"]),
         ("lr = 0.1", baseline_accs),
         ("lr = 0.5", results["lr=0.5"]["val_accuracies"])],
        "Learning rate (hidden = 128, batch = 64)", "exp_learning_rate.png")

    plot_accuracy_comparison(
        [("hidden = 32", results["hidden_size=32"]["val_accuracies"]),
         ("hidden = 128", baseline_accs),
         ("hidden = 512", results["hidden_size=512"]["val_accuracies"])],
        "Hidden layer size (lr = 0.1, batch = 64)", "exp_hidden_size.png")

    plot_long_training(results["epochs=30"],
                       "Long training: 30 epochs (lr = 0.1, hidden = 128)", "exp_30_epochs.png")

    print(f"Figures saved to {FIGURES_DIR}")
