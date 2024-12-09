# Segmentation
grid_size=(16,16)
def segment_image(image, grid_size=grid_size):
        h, w = image.shape
        rows, cols = grid_size
        segment_height = h // rows
        segment_width = w // cols
        
        segments = []
        for r in range(rows):
            for c in range(cols):
                segment = image[r * segment_height:(r + 1) * segment_height, c * segment_width:(c + 1) * segment_width]
                segments.append(segment)
        return segments

def segment_dataset(images):
    segmented_images = []
    for img, filename, original_size in images:
        segmented_img = segment_image(img, grid_size=grid_size)
        segmented_images.append(segmented_img)
    return segmented_images