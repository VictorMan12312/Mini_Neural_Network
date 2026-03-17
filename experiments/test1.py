from interfaces.api import APIModel
from core.layers import Dense, ReLU, Sigmoid
from utils.data_loader import generate_synthetic, train_test_split
from utils.visualization import plot_loss
from utils.metrics import accuracy
def run():
    print(">> Setting up Synthetic Binary Classification Experiment")
    X, y = generate_synthetic(samples=1000)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    layers = [
        Dense(2, 16),
        ReLU(),
        Dense(16, 1),
        Sigmoid()
    ]
    model = APIModel(layers, loss_type='bce')
    model.train(X_train, y_train, optimizer_name="gd", epochs=150, lr=0.1)
    preds = model.forward(X_test)
    print(f">> Final Accuracy: {accuracy(preds, y_test) * 100:.2f}%")
    plot_loss(model.history, title="Synthetic Data - GD Convergence")
if __name__ == "__main__":
    run()