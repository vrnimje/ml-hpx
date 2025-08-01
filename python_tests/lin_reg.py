from ml_hpx import LinearRegression
import pandas as pd

df = pd.read_csv('./datasets/linear_regressor_dataset_10000.csv')
X = df['x']
Y = df['y']

lin_reg = LinearRegression(5000, 1e-5)

print(lin_reg.fit(X.values, Y.values))
print(lin_reg.predict(79.8690852138018))
