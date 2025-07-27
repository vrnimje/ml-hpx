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
    parameter_pairs = [(2.577991, -0.136139), (2.514959, 3.057930),
                       (2.530195, 2.043939), (2.523242, 2.506657)]  # List of (W, B) pairs
    plot_linear_regressor(dataset_path, parameter_pairs)
