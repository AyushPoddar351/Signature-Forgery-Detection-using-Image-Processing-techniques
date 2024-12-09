import cv2
import numpy as np

 # Pre-processing 
def preprocess_signature(image, resize_dim=(512, 512)):
        """
        Enhanced preprocessing for signature images.
        - Preserves the structure of the signature while standardizing the input format.
        """
        # Convert to grayscale (if not already)
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Bilateral Filter for noise reduction
        filtered_image = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

        # Adaptive Thresholding for binarization
        binary_image = cv2.adaptiveThreshold(
            filtered_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, blockSize=11, C=2
        )

        # Extract the signature region using contours
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get bounding box of the largest contour
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))

            # Add padding to the bounding box
            padding = 10
            x = max(0, x - padding)
            y = max(0, y - padding)
            w += 2 * padding
            h += 2 * padding

            cropped_image = binary_image[y:y+h, x:x+w]
        else:
            # If no contours found, use the whole image
            cropped_image = binary_image

        # Resize with aspect ratio preservation
        h, w = cropped_image.shape
        scale = min(resize_dim[0] / h, resize_dim[1] / w)
        new_h, new_w = int(h * scale), int(w * scale)
        resized_image = cv2.resize(cropped_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Create a blank canvas and center the resized image
        final_image = np.ones(resize_dim, dtype=np.uint8) * 255  # White background
        y_offset = (resize_dim[0] - new_h) // 2
        x_offset = (resize_dim[1] - new_w) // 2
        final_image[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_image

        return final_image

# Preprocess dataset
def preprocess_dataset(images):
        """
        Preprocesses a dataset of signature images.
        - Applies preprocessing to each image in the dataset.
        """
        preprocessed_images = []
        for img, filename, original_size in images:
            preprocessed_img = preprocess_signature(img)
            resize = preprocessed_img.shape
            preprocessed_images.append((preprocessed_img, filename, resize))
        return preprocessed_images