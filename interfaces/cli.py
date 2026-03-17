import argparse
import sys
from interfaces.api import APIModel
from core.layers import Dense, ReLU, Sigmoid
from utils.data_loader import generate_synthetic, load_csv, train_test_split
from utils.metrics import accuracy
def run_cli():
    parser = argparse.ArgumentParser(description="Train a mini_nn model via CLI.")
    parser.add_argument("--dataset", type=str, default="synthetic", help="Path to csv dataset or 'synthetic'")
    parser.add_argument("--optimizer", type=str, default="gd", choices=["gd", "newton"], help="Optimizer to use")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    
    # Need to check sys.argv directly to avoid problems when imported vs direct execution
    args = parser.parse_args(sys.argv[1:])
    
    if args.dataset == "synthetic":
        print("Using generated synthetic dataset...")
        X, y = generate_synthetic(500)
    else:
        try:
            print(f"Loading CSV dataset: {args.dataset}")
            X, y = load_csv(args.dataset)
        except Exception as e:
            print(f"Error loading {args.dataset}: {e}")
            return
            
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    # Build typical binary classification network
    layers = [
        Dense(X_train.shape[1], 16),
        ReLU(),
        Dense(16, 1),
        Sigmoid()
    ]
    
    model = APIModel(layers, loss_type="bce")
    
    print(f"\n--- Starting Training ({args.optimizer.upper()}) ---")
    model.train(X_train, y_train, optimizer_name=args.optimizer, epochs=args.epochs, lr=args.lr)
    
    # Evaluate
    preds = model.forward(X_test)
    acc = accuracy(preds, y_test)
    print(f"\n[Result] Final Test Accuracy: {acc * 100:.2f}%\n")
if __name__ == "__main__":
    run_cli()