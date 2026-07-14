import re
import matplotlib.pyplot as plt
import os

LOG_PATH = "training_log.txt"
OUTPUT_PATH = "training_progress.png"

def plot_progress():
    if not os.path.exists(LOG_PATH):
        print(f"Log file not found at {LOG_PATH}")
        return

    epochs = []
    train_losses = []
    val_losses = []
    accuracies = []

    with open(LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if "Epoch" in line and "|" in line:
                try:
                    # Parse Epoch
                    epoch_match = re.search(r"Epoch\s+(\d+)/", line)
                    if not epoch_match:
                        continue
                    epoch = int(epoch_match.group(1))

                    # Parse Accuracy
                    acc_match = re.search(r"Acc:\s*([\d\.]+)%", line) or re.search(r"Train Acc:\s*([\d\.]+)%", line)
                    acc = float(acc_match.group(1)) if acc_match else 0.0

                    # Parse Train Loss
                    t_loss_match = re.search(r"Train Loss:\s*([\d\.]+)", line) or re.search(r"Loss:\s*([\d\.]+)", line)
                    t_loss = float(t_loss_match.group(1)) if t_loss_match else 0.0

                    # Parse Val Loss
                    v_loss_match = re.search(r"Val Loss:\s*([\d\.]+)", line)
                    v_loss = float(v_loss_match.group(1)) if v_loss_match else t_loss  # Fallback to train loss if no val loss

                    epochs.append(epoch)
                    train_losses.append(t_loss)
                    val_losses.append(v_loss)
                    accuracies.append(acc)
                except Exception as e:
                    print(f"Error parsing line: {line.strip()} -> {e}")

    if not epochs:
        print("No epoch data found in log yet. Wait for at least one epoch to complete.")
        return

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Plot
    ax1.plot(epochs, train_losses, 'b.-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r.-', label='Val Loss')
    ax1.set_title('Model Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()

    # Accuracy Plot
    ax2.plot(epochs, accuracies, 'g.-', label='Accuracy (%)')
    ax2.set_title('Model Accuracy Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150)
    print(f"Graph successfully saved as {OUTPUT_PATH}")

if __name__ == "__main__":
    plot_progress()
