import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Load all images from the directory
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

# Divide the images into training (80%) and testing (20%)
def split_images(images):
    n_train = int(0.8 * len(images))
    train_images = images[:n_train]
    test_images = images[n_train:]
    return train_images, test_images

# Extract RGB channels and calculate ratio = (B - R) / (B + R)
def calculate_ratio(image):
    R = image[:, :, 0].astype(float)
    G = image[:, :, 1].astype(float)
    B = image[:, :, 2].astype(float)
    ratio = np.divide((B - R), (B + R), out=np.zeros_like(B), where=(B + R) != 0)
    return ratio

# Train KMeans clustering model
def train_kmeans_on_images(train_images, n_clusters=4):
    all_ratios = []
    
    for image in train_images:
        ratio = calculate_ratio(image)
        all_ratios.append(ratio.flatten())
    
    # All images ratios into a single array for clustering
    all_ratios = np.concatenate(all_ratios).reshape(-1, 1)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(all_ratios)
    
    return kmeans

# Assign custom cluster names
def assign_cluster_names(kmeans):
    cluster_names = {0: "MC", 1: "CS", 2: "OV", 3: "PC"}
    return cluster_names

# Test the trained model on the test dataset
def test_kmeans_on_images(test_images, kmeans):
    test_results = []
    
    for image in test_images:
        ratio = calculate_ratio(image)
        ratio_reshaped = ratio.flatten().reshape(-1, 1)
        predictions = kmeans.predict(ratio_reshaped)
        test_results.append(predictions.reshape(ratio.shape))
    
    return np.array(test_results)

# Visualize the results
def visualize_clusters(test_images, test_results, cluster_names):
    n_images = len(test_images)
    
    for i in range(n_images):
        image = test_images[i]
        clustered_result = test_results[i]
        
        h, w = clustered_result.shape
        clustered_image = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Map each cluster to a specific color for visualization
        cluster_colors = {
            0: [192, 192, 192],    # Red for Cluster A
            1: [31, 119, 180],    # Green for Cluster B
            2: [128, 128, 128],    # Blue for Cluster C
            3: [176, 196, 222],  # Yellow for Cluster D
        }
        
        for cluster_id, color in cluster_colors.items():
            clustered_image[clustered_result == cluster_id] = color

        # Prepare the legend for the clusters
        legend_patches = [
            mpatches.Patch(color=np.array(color) / 255.0, label=cluster_names[cluster_id])
            for cluster_id, color in cluster_colors.items()
        ]

        # Plot the original and clustered images side by side
        plt.figure(figsize=(10, 5))
        
        # Plot original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.axis('off')  # Remove axis numbers and ticks
        
        # Plot clustered image with legend
        plt.subplot(1, 2, 2)
        plt.imshow(clustered_image)
        plt.title("Clustered Image")
        plt.axis('off')  # Remove axis numbers and ticks
        
        # Add legend
        plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.2, 1))
        
        plt.show()

# Image path
folder_path = r'C:\Users\berhaned\OneDrive - SINTEF\Berhane_SIN_Industri\Jupyter_Python_SINTEF\SEP_2024\ghi-sky-mapper\ASI_imges'
current_path = os.getcwd()
folder_path = f'{current_path}/ASI_imges/'
# Load images from the directory
images = load_images_from_directory(folder_path)

# Split the images into training and testing datasets
train_images, test_images = split_images(images)

# Calculate the ratio and train KMeans on the training dataset
kmeans_model = train_kmeans_on_images(train_images)

# Assign custom cluster names
cluster_names = assign_cluster_names(kmeans_model)

# Test the KMeans model on the test dataset
test_results = test_kmeans_on_images(test_images, kmeans_model)

# Visualize the test results
visualize_clusters(test_images, test_results, cluster_names)




def mapping(clustered_result, cluster_names, ghi_value):
    """
    Maps clusters of sky conditions to an irradiation map based on transmittance values and GHI.

    """
    
    # Define transmittance values for each cluster
    transmittance_values = {
        "MC": 0.4,  # Mostly cloudy
        "CS": 0.9,  # Clear sky, high transmittance
        "OV": 0.1,  # Overcast sky, low transmittance
        "PC": 0.7   # Partly cloudy
    }
    
    # Create a map of transmittance based on the clusters
    h, w = clustered_result.shape
    transmittance_map = np.zeros((h, w))
    
    for cluster_id, cluster_name in cluster_names.items():
        transmittance_map[clustered_result == cluster_id] = transmittance_values[cluster_name]
    
    # Multiply the transmittance by the GHI value to compute the irradiation map
    irradiation_map = transmittance_map * ghi_value
    
    return irradiation_map


# This has to be connected with pvlib library

ghi_value = 1000  # Example GHI value in W/m²

# Define the colormap for the cluster visualization
cluster_colors = {
    0: [192, 192, 192],    # Red for Cluster MC
    1: [31, 119, 180],    # Green for Cluster CS
    2: [128, 128, 128],    # Blue for Cluster OV
    3: [176, 196, 222],  # Yellow for Cluster PC
}


# Loop over each image in the test_results
for i, result in enumerate(test_results):
    # Original image
    original_image = test_images[i]

    # Mapping function
    irradiation_map = mapping(result, cluster_names, ghi_value)

    # Clustered image with colors
    h, w = result.shape
    clustered_image = np.zeros((h, w, 3), dtype=np.uint8)
    for cluster_id, color in cluster_colors.items():
        clustered_image[result == cluster_id] = color

    # Legend patches
    legend_patches = [
        mpatches.Patch(color=np.array(color) / 255.0, label=cluster_names[cluster_id])
        for cluster_id, color in cluster_colors.items()
    ]

    # The original image, clustered image, and irradiation map side by side
    plt.figure(figsize=(18, 6))

    # Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(original_image)
    plt.title("Original Image")
    plt.axis('off')

    # Clustered Image with Legend
    plt.subplot(1, 3, 2)
    plt.imshow(clustered_image)
    plt.title("Clustered Image")
    plt.axis('off')
    plt.legend(handles=legend_patches, loc='upper right', bbox_to_anchor=(1.2, 1))

    # Irradiation Mapped Image
    plt.subplot(1, 3, 3)
    plt.imshow(irradiation_map, cmap='hot_r') 
    plt.title("Irradiation Map")
    plt.colorbar(label="Irradiation (W/m²)")
    plt.axis('off')

    plt.show()
