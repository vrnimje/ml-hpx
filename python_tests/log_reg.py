from ml_hpx import LogisticRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import sklearn.linear_model
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('./datasets/logistic_regression_dataset_10000.csv')

X = df[['x']]
y = df['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

train = list(zip(X_train.flatten().tolist(), y_train))

log_reg = LogisticRegression(5000, 0.005, 123)
print(f"Train accuracy: {log_reg.train(train)}")

y_pred = log_reg.predict(X_test.flatten().tolist())

# Evaluate the model
print("Test accuracy:", accuracy_score(y_test, y_pred))

# Create and fit the logistic regression model
model = sklearn.linear_model.LogisticRegression()
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
