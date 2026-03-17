import matplotlib.pyplot as plt
def plot_loss(history: list, title: str = "Training Loss"):
    """Plots the loss curve of a single model."""
    plt.figure(figsize=(8, 5))
    plt.plot(history, label="Loss Curve", color="blue", linewidth=2)
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()
def plot_optimizer_comparison(histories: list, labels: list):
    """Plots and compares loss curves from multiple optimizers."""
    plt.figure(figsize=(8, 5))
    for hist, label in zip(histories, labels):
        plt.plot(hist, label=label, linewidth=2)
        
    plt.title("Optimizer Convergence Comparison")
    plt.xlabel("Epochs")
    plt.ylabel("Loss value")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.show()