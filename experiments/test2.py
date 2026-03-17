from interfaces.api import APIModel
from core.layers import Dense, ReLU, Sigmoid
from utils.data_loader import generate_synthetic
from utils.visualization import plot_optimizer_comparison
def run():
    print(">> Experiment: Comparing GD vs Newton on Titanic (Mock Data)")
    
    # In production, swap with `load_csv("titanic_cleaned.csv")`
    X, y = generate_synthetic(400)
    
    print("\nTraining Model 1: Gradient Descent")
    model_gd = APIModel([Dense(2,8), ReLU(), Dense(8,1), Sigmoid()], 'bce')
    model_gd.train(X, y, optimizer_name='gd', epochs=80, lr=0.1, verbose=False)
    
    # Newton usually relies on smaller learning rates for highly approximated Hessians.
    print("Training Model 2: Newton Method")
    model_nt = APIModel([Dense(2,8), ReLU(), Dense(8,1), Sigmoid()], 'bce')
    model_nt.train(X, y, optimizer_name='newton', epochs=80, lr=0.5, verbose=False)
    print(">> Processing Plot Comparison...")
    plot_optimizer_comparison(
        [model_gd.history, model_nt.history],
        ["Gradient Descent (lr=0.1)", "Newton Method (lr=0.5)"]
    )
if __name__ == "__main__":
    run()