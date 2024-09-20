import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Step 1: Accept folder path and load all images from the directory
def load_images_from_directory(folder_path):
    if not os.path.isdir(folder_path):
        raise ValueError(f"The provided folder path does not exist: {folder_path}")
    
    loaded_images = []
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image at {image_path}, skipping.")
                continue
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            loaded_images.append(image_rgb)
        else:
            print(f"File {filename} is not a valid image file, skipping.")
    
    if not loaded_images:
        raise ValueError(f"No valid images were found in the folder: {folder_path}")
    
    return loaded_images

# Step 2: Divide the images into training (80%) and testing (20%)
def split_images(images):
    n_train = int(0.8 * len(images))
    train_images = images[:n_train]
    test_images = images[n_train:]
    return train_images, test_images

# Step 3: Extract RGB channels and calculate ratio = (B - R) / (B + R)
def calculate_ratio(image):
    R = image[:, :, 0].astype(float)
    G = image[:, :, 1].astype(float)
    B = image[:, :, 2].astype(float)

    # Avoid division by zero
    ratio = np.divide((B - R), (B + R), out=np.zeros_like(B), where=(B + R) != 0)
    return ratio

# Step 4: Train KMeans clustering model
def train_kmeans_on_images(train_images, n_clusters=4):
    all_ratios = []
    
    for image in train_images:
        ratio = calculate_ratio(image)
        all_ratios.append(ratio.flatten())
    
    # Combine all images' ratios into a single array for clustering
    all_ratios = np.concatenate(all_ratios).reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(all_ratios)
    
    return kmeans

# Step 5: Assign custom cluster names
def assign_cluster_names(kmeans):
    cluster_names = {0: "Cluster A", 1: "Cluster B", 2: "Cluster C", 3: "Cluster D"}
    return cluster_names

# Step 6: Test the trained model on the test dataset
def test_kmeans_on_images(test_images, kmeans):
    test_results = []
    
    for image in test_images:
        ratio = calculate_ratio(image)
        ratio_reshaped = ratio.flatten().reshape(-1, 1)
        predictions = kmeans.predict(ratio_reshaped)
        test_results.append(predictions.reshape(ratio.shape))
    
    return test_results

# Step 7: Visualize the clustering results using plots
def visualize_clusters(test_images, test_results, cluster_names):
    n_images = len(test_images)
    
    for i in range(n_images):
        image = test_images[i]
        clustered_result = test_results[i]
        
        h, w = clustered_result.shape
        clustered_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Map each cluster to a specific color for visualization
        cluster_colors = {
            0: [255, 0, 0],    # Red for Cluster A
            1: [0, 255, 0],    # Green for Cluster B
            2: [0, 0, 255],    # Blue for Cluster C
            3: [255, 255, 0],  # Yellow for Cluster D
        }
        
        for cluster_id, color in cluster_colors.items():
            clustered_image[clustered_result == cluster_id] = color

        # Plot the original and clustered images side by side
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        
        plt.subplot(1, 2, 2)
        plt.imshow(clustered_image)
        plt.title("Clustered Image")
        
        plt.show()

# Main function to execute the complete workflow
folder_path = r'C:\Users\berhaned\OneDrive - SINTEF\Berhane_SIN_Industri\Jupyter_Python_SINTEF\SEP_2024\ghi-sky-mapper\ASI_imges'
# Step 1: Load images from the directory
images = load_images_from_directory(folder_path)

# Step 2: Split the images into training and testing datasets
train_images, test_images = split_images(images)

# Step 3 & 4: Calculate the ratio and train KMeans on the training dataset
kmeans_model = train_kmeans_on_images(train_images)

# Step 5: Assign custom cluster names
cluster_names = assign_cluster_names(kmeans_model)

# Step 6: Test the KMeans model on the test dataset
test_results = test_kmeans_on_images(test_images, kmeans_model)

# Step 7: Visualize the test results
visualize_clusters(test_images, test_results, cluster_names)