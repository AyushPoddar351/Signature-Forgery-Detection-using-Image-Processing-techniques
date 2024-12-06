# Importing Libraries
import cv2
import numpy as np
import os
import json
import preprocessing as pp, segmentation as sg, comparison as cm

def forgery(img, path):
        
    test_signature_path = img # Update this to your test image file path
    
    # Helper function to load images and display their sizes
    def load_images_from_folder(folder_path):
        images = []
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append((img, filename, img.shape))  # Store image, filename, and size
        return images
    
    # Load original and test signatures
    original_signatures = load_images_from_folder(path)
    test_signature = cv2.imread(test_signature_path, cv2.IMREAD_GRAYSCALE)

    preprocessed_original_signatures = pp.preprocess_dataset(original_signatures)
    preprocessed_test_signature = pp.preprocess_signature(test_signature)

    
    # Segmentation
    grid_size=(16,16)
    # Segment all preprocessed original signatures with grids
    segmented_signatures_16x16 = sg.segment_dataset(preprocessed_original_signatures)

    # Segment the preprocessed test signature with grids
    segmented_test_signature_16x16 = sg.segment_image(preprocessed_test_signature, grid_size=grid_size)

    
    #Test Comparison
    comparison_results = cm.compare_signature(
            original_segments_list=segmented_signatures_16x16,
            test_segments=segmented_test_signature_16x16,  # Replace for meaningful comparisons
            weights=None,
            skip_identical=True
        )

    mse_test = np.mean([result['mse_avg'] for result in comparison_results])
    ssim_test = np.mean([result['ssim_avg'] for result in comparison_results])
    temp_test = np.mean([result['template_avg'] for result in comparison_results])
    hist_test = np.mean([result['histogram_avg'] for result in comparison_results])
    hogs_test = np.mean([result['hog_avg'] for result in comparison_results])
    nmi_test = np.mean([result['nmi_avg'] for result in comparison_results])

    test_scores = {
                "mse": mse_test,
                "ssim": ssim_test,
                "template": temp_test,
                "histogram": hist_test,
                "hog": hogs_test,
                "nmi": nmi_test,
            }


    # Confidence Score Calculation
    diff = {}
    # Read from the JSON file
    with open('Flask app/scores.json', 'r') as json_file:
        scores = json.load(json_file)
    for metric in test_scores:
        diff[metric] = scores[metric] - test_scores[metric]

    # Check if three or more values in the test dictionary are positive
    count = sum(1 for value in diff.values() if value > 0)
    percent = (count/6.0)*100
    result = {}
    # Determine if the signature is forged
    if count > 3:
        result["Forged"] = percent
    else:
        result["Genuine"] = 100.0 - percent    
    return result