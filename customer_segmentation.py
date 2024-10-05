import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset (update the path if necessary)
data_path = 'C:/Users/charu/machine_learning_intership/task2/Mall_Customers.csv'
customer_data = pd.read_csv(data_path)

# Display the first few rows of the dataset to verify the data structure
print(customer_data.head())

# Step 1: Data Preprocessing
# Select relevant features for clustering (Age, Annual Income, Spending Score)
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
X = customer_data[features]

# Step 2: Normalize the data (optional, depending on the distribution of the features)
# X = (X - X.mean()) / X.std()

# Step 3: Determine the optimal number of clusters using the elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Step 4: Fit the K-means model with the optimal number of clusters (choose k based on elbow curve)
optimal_k = 5  # You can change this based on the elbow curve
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
customer_data['Cluster'] = kmeans.fit_predict(X)

# Step 5: Visualize the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=customer_data, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Cluster', palette='Set1')
plt.title('Customer Segmentation')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend(title='Cluster')
plt.show()

# Step 6: Save the clustered data to a new CSV file
output_path = 'customer_segmented_data.csv'
customer_data.to_csv(output_path, index=False)
print(f"Clustered data saved to {output_path}")

