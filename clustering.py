import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from matplotlib.patches import Patch

def segment_image(image_path, clusters):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image. Please check the path.")
        return None, None

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Define criteria for KMeans clustering
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # Apply KMeans clustering to segment the image
    _, labels, centers = cv2.kmeans(pixel_values, clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Convert centers to 8-bit values and flatten the label array
    centers = np.uint8(centers)
    segmented_image = centers[labels.flatten()]

    # Reshape back to the original image dimensions
    segmented_image = segmented_image.reshape(image_rgb.shape)

    # Draw contours around the segmented regions
    mask = np.zeros_like(segmented_image)
    for i in range(clusters):
        mask[labels.reshape(image_rgb.shape[:2]) == i] = centers[i]
        contours, _ = cv2.findContours((mask[..., 0] == centers[i][0]).astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(segmented_image, contours, -1, (0, 0, 0), 2)  # Draw thick contours

    return segmented_image, centers

def get_cluster_labels(centers):
    # Sort cluster centers based on their brightness to label clouds
    brightness = np.sum(centers, axis=1)
    sorted_indices = np.argsort(brightness)
    labels = ['Clear', 'Partly Cloudy', 'Cloudy', 'Overcast']

    # Map sorted indices to labels, adjusting based on cluster size
    cluster_labels = {index: labels[min(i, len(labels) - 1)] for i, index in enumerate(sorted_indices)}
    return cluster_labels

def plot_comparison(image_path, save_path):
    # Load the original image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Define the number of clusters to test
    cluster_values = [3, 6, 9]  # Adjust as needed for more comparisons

    # Create a plot for comparison
    fig, axarr = plt.subplots(1, len(cluster_values) + 1, figsize=(20, 10))
    axarr[0].imshow(image_rgb)
    axarr[0].set_title('Original Image')
    axarr[0].axis('off')

    all_patches = []  # For creating the legend

    # Loop through different cluster values and display results
    for i, k in enumerate(cluster_values):
        segmented_image, centers = segment_image(image_path, k)
        cluster_labels = get_cluster_labels(centers)

        # Create patches for legend
        patches = [Patch(color=np.array(center)/255, label=f'{cluster_labels[j]}') for j, center in enumerate(centers)]
        all_patches.extend(patches)

        # Plot each segmentation
        axarr[i + 1].imshow(segmented_image)
        axarr[i + 1].set_title(f'Segmented with {k} Clusters')
        axarr[i + 1].axis('off')

    # Add a legend to the main figure without overlapping the images
    fig.legend(handles=all_patches, loc='upper center', ncol=len(cluster_values), fontsize='small', frameon=False)

    # Save the plot
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Comparison plot saved as {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py path_to_image")
    else:
        image_path = sys.argv[1]
        # Define the save path for the output plot
        save_path = os.path.join(os.path.dirname(image_path), "segmentation_comparison.png")
        plot_comparison(image_path, save_path)
