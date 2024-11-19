# Histogram-Equalization
```python
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
from google.colab import files
from IPython.display import display, Image

# Function to upload image
def upload_image():
    uploaded = files.upload()
    for filename in uploaded.keys():
        print(f"Uploaded file: {filename}")
        return filename

# Step 1: Upload image
print("Please upload an image file:")
image_path = upload_image()

# Step 2: Load the image
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

if image is None:
    print("Error: Unable to read the uploaded file. Please upload a valid image.")
else:
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Initialize histogram and cumulative histogram
    h = [0] * 256
    h2 = [0] * 256

    # Compute the histogram
    height, width = gray.shape
    for i in range(height):
        for j in range(width):
            intensity = gray[i, j]
            h[intensity] += 1

    # Calculate the cumulative histogram and normalized values
    c = np.cumsum(h)
    c_norm = c / c[-1] * 255  # Normalize to the range [0, 255]
    c_norm = c_norm.astype(int)

    # Apply histogram equalization
    equalized_image = np.zeros_like(gray)
    for i in range(height):
        for j in range(width):
            equalized_image[i, j] = c_norm[gray[i, j]]
            h2[equalized_image[i, j]] += 1

    # Plot the original and equalized histograms
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.title("Original Image")
    plt.imshow(gray, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Equalized Image")
    plt.imshow(equalized_image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Original Histogram")
    plt.bar(range(256), h, color='gray')

    plt.subplot(2, 2, 4)
    plt.title("Equalized Histogram")
    plt.bar(range(256), h2, color='gray')

    plt.tight_layout()
    plt.show()

    # Save the equalized image
    output_path = "equalized_image.png"
    cv2.imwrite(output_path, equalized_image)
    print(f"Equalized image saved as: {output_path}")

    # Display saved image
    display(Image(output_path))
```
Output :
```

```
