import pandas as pd
import matplotlib.pyplot as plt
import sys

# Function to plot the dataset and test fits
def plot_linear_regressor(dataset_path, parameter_pairs):
    data = pd.read_csv(dataset_path)
    x = data['x'].values
    y = data['y'].values

    plt.figure(figsize=(20, 12))
    plt.scatter(x, y, color='blue', alpha=0.5, label='Data Points', s=10)

    for W, B in parameter_pairs:
        y_pred = W * x + B
        plt.plot(x, y_pred, label=f'W={W}, B={B}')

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regressor Dataset with Multiple Fits')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    dataset_path = sys.argv[1]  # Path to the dataset
    parameter_pairs = [(2.530294, 3.038263), (2.558153, 0.183712), (2.558576, 0.155537), (2.559148, 0.117489)]  # List of (W, B) pairs
    plot_linear_regressor(dataset_path, parameter_pairs)
