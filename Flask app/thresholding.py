# Importing Libraries
import cv2
import numpy as np
import os
import json
import preprocessing as pp, segmentation as sg, comparison as cm

original_scores={}

def threshold(path):
        
    # Path to your dataset
    original_signatures_path = path  # Update this to your folder path
    
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
    original_signatures = load_images_from_folder(original_signatures_path)
    preprocessed_original_signatures = pp.preprocess_dataset(original_signatures)
   

    # Segment all preprocessed original signatures with grids
    segmented_signatures_16x16 = sg.segment_dataset(preprocessed_original_signatures)

    # Thresholding
    mse_scores, ssim_scores, temp_scores, hist_scores, hog_scores, nmi_scores=[],[],[],[],[],[]

    for i in range(len(segmented_signatures_16x16)):
        print(f"Comparison with Original Signature {i + 1}")
        comparison_results = cm.compare_signature(
            original_segments_list=segmented_signatures_16x16,
            test_segments=segmented_signatures_16x16[i],  # Replace for meaningful comparisons
            weights=None,
            skip_identical=True
        )

        mse = np.mean([result['mse_avg'] for result in comparison_results])
        ssim = np.mean([result['ssim_avg'] for result in comparison_results])
        temp = np.mean([result['template_avg'] for result in comparison_results])
        hist = np.mean([result['histogram_avg'] for result in comparison_results])
        hogs = np.mean([result['hog_avg'] for result in comparison_results])
        nmi = np.mean([result['nmi_avg'] for result in comparison_results])

        print(
        f"Average Scores:\n"
        f"MSE: {mse:.4f}, SSIM: {ssim:.4f}, "
        f"Template: {temp:.4f}, Histogram: {hist:.4f}, HOG: {hogs:.4f}, NMI: {nmi:.4f}\n"
        )

        mse_scores.append(mse)
        ssim_scores.append(ssim)
        temp_scores.append(temp)
        hist_scores.append(hist)
        hog_scores.append(hogs)
        nmi_scores.append(nmi)

    mse_org_16x16 = np.mean([mse for mse in mse_scores])
    ssim_org_16x16 = np.mean([ssim for ssim in ssim_scores])
    temp_org_16x16 = np.mean([temp for temp in temp_scores])
    hist_org_16x16 = np.mean([hist for hist in hist_scores])
    hog_org_16x16 = np.mean([hogs for hogs in hog_scores])
    nmi_org_16x16 = np.mean([nmi for nmi in nmi_scores])

    scores = {
                "mse": mse_org_16x16,
                "ssim": ssim_org_16x16,
                "template": temp_org_16x16,
                "histogram": hist_org_16x16,
                "hog": hog_org_16x16,
                "nmi": nmi_org_16x16,
            }
    
    scores = {key: float(value) if isinstance(value, np.float32) else value for key, value in scores.items()}

    # Write to a JSON file
    with open('Flask app/scores.json', 'w') as json_file:
        json.dump(scores, json_file)
