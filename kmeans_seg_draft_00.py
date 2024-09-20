import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Step 1: Accept folder path and load image
def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load the image at {image_path}")
    # Convert BGR (default in OpenCV) to RGB for consistency
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# Step 2: Divide the image into training (80%) and testing (20%)
def split_image(image):
    h, w, _ = image.shape
    # Reshape to (num_pixels, 3) where 3 is for RGB
    pixels = image.reshape(-1, 3)

    # Use sklearn train_test_split to split into 80% training and 20% testing
    train_pixels, test_pixels = train_test_split(pixels, test_size=0.2, shuffle=False)
    return train_pixels, test_pixels

# Step 3: Extract RGB channels and calculate the ratio = (B - R) / (B + R)
def calculate_ratio(pixels):
    R = pixels[:, 0].astype(float)
    G = pixels[:, 1].astype(float)
    B = pixels[:, 2].astype(float)

    # Avoid division by zero
    ratio = np.divide((B - R), (B + R), out=np.zeros_like(B), where=(B + R) != 0)
    return ratio

# Step 4: Train KMeans clustering model
def train_kmeans(ratio, n_clusters=4):
    # Reshape the ratio array to 2D to fit KMeans (num_pixels, 1)
    ratio_reshaped = ratio.reshape(-1, 1)

    # Train the KMeans clustering algorithm
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(ratio_reshaped)
    return kmeans

# Step 5: Assign custom cluster names
def assign_cluster_names(kmeans):
    cluster_names = {0: "Cluster A", 1: "Cluster B", 2: "Cluster C", 3: "Cluster D"}
    return [cluster_names[label] for label in kmeans.labels_]

# Step 6: Test the trained model
def test_model(kmeans, test_ratio):
    test_ratio_reshaped = test_ratio.reshape(-1, 1)
    predictions = kmeans.predict(test_ratio_reshaped)
    return predictions

# Step 7: Visualize clusters using plots
def visualize_clusters(image, train_pixels, test_pixels, train_labels, test_labels):
    h, w, _ = image.shape
    
    # Reshape the labels back to the image shape
    train_image_labels = np.zeros((train_pixels.shape[0], 3), dtype=int)
    test_image_labels = np.zeros((test_pixels.shape[0], 3), dtype=int)

    cluster_colors = {
        0: [255, 0, 0],    # Red for Cluster A
        1: [0, 255, 0],    # Green for Cluster B
        2: [0, 0, 255],    # Blue for Cluster C
        3: [255, 255, 0],  # Yellow for Cluster D
    }

    # Map the cluster labels to colors for visualization
    for idx, label in enumerate(train_labels):
        train_image_labels[idx] = cluster_colors[label]

    for idx, label in enumerate(test_labels):
        test_image_labels[idx] = cluster_colors[label]

    # Reshape train and test labeled images back to original dimensions
    train_image = train_image_labels.reshape((int(h * 0.8), w, 3))
    test_image = test_image_labels.reshape((int(h * 0.2), w, 3))

    # Plot the original image, training clusters, and testing clusters
    fig, axs = plt.subplots(1, 3, figsize=(20, 10))
    axs[0].imshow(image)
    axs[0].set_title("Original Image")
    
    axs[1].imshow(train_image)
    axs[1].set_title("Training Clusters")
    
    axs[2].imshow(test_image)
    axs[2].set_title("Testing Clusters")

    plt.show()

# Main function to process the image and run the steps
def main(image_path):
    # Load the image
    image = load_image(image_path)

    # Split the image into training and testing sets
    train_pixels, test_pixels = split_image(image)

    # Calculate the ratio for training and testing sets
    train_ratio = calculate_ratio(train_pixels)
    test_ratio = calculate_ratio(test_pixels)

    # Train KMeans on training data
    kmeans = train_kmeans(train_ratio)

    # Assign custom cluster names to the trained model
    train_labels = kmeans.predict(train_ratio.reshape(-1, 1))
    
    # Test the model on the test data
    test_labels = test_model(kmeans, test_ratio)

    # Visualize the results
    visualize_clusters(image, train_pixels, test_pixels, train_labels, test_labels)

# Provide the image path and run the program
if __name__ == "__main__":
    image_path = "path_to_your_image.jpg"
    main(image_path)
