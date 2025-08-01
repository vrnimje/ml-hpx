from ml_hpx import KNearestNeighbours
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

df = pd.read_csv('./datasets/classified_points_dataset.csv')

X = df[['x', 'y']]
y = df['class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNearestNeighbours(k=3)

knn.fit(X_train.tolist(), y_train.values)

y_pred = knn.predict(X_test.tolist())

print("ML-HPX accuracy", accuracy_score(y_pred, y_test))

knn_sklearn = KNeighborsClassifier(n_neighbors=3)

knn_sklearn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

print("Sklearn accuracy", accuracy_score(y_pred, y_test))
