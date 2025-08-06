from ml_hpx import KMeansClustering
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pandas as pd
from time import perf_counter

# Load dataset
df = pd.read_csv('./datasets/kmeans_test_dataset.csv')

X = df[['x', 'y']]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

### scikit-learn KMeans ###
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)

start_kmeans = perf_counter()
kmeans.fit(X_scaled)
end_kmeans = perf_counter()

# Add cluster labels to DataFrame
df['cluster'] = kmeans.labels_

# Print performance
print(f"Sklearn KMeans time: {end_kmeans - start_kmeans:.4f} seconds")
print(f"Inertia: {kmeans.inertia_:.4f}")

kmeans_hpx = KMeansClustering(k=5)

start_hpx = perf_counter()
sse = kmeans_hpx.fit(X_scaled.tolist())
end_hpx = perf_counter()

print(f"ML-HPX KMeans time: {end_hpx - start_hpx:.4f} seconds")
print(f"Inertia: {sse:.4f}")
