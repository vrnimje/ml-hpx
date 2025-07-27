from ml_hpx import LogisticRegression
import pandas as pd

df = pd.read_csv('./datasets/logistic_regression_dataset_10000.csv')

X = df['x']
Y = df['class']

data = list(zip(X, Y))

log_reg = LogisticRegression(4000, 0.001)
print(log_reg.train(data))
