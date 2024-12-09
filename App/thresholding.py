import cv2
import numpy as np
import os
import json
import preprocessing as pp, segmentation as sg, comparison as cm

original_scores = {}

def threshold(path, progress_callback=None, message_callback=None):
    original_signatures_path = path

    def log_message(message):
        if message_callback:
            message_callback(message)

    def load_images_from_folder(folder_path):
        images = []
        for filename in os.listdir(folder_path):
            img_path = os.path.join(folder_path, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append((img, filename, img.shape))
        return images

    # Step 1: Load images (0% to 10%)
    log_message("Loading original signatures...")
    original_signatures = load_images_from_folder(original_signatures_path)
    log_message(f"Loaded {len(original_signatures)} original signatures.")
    if progress_callback:
        progress_callback(10)

    # Step 2: Preprocess and Segment images (10% to 30%)
    log_message("Preprocessing images...")
    preprocessed_original_signatures = pp.preprocess_dataset(original_signatures)
    log_message("Preprocessing completed.")
    
    segmented_signatures_16x16 = sg.segment_dataset(preprocessed_original_signatures)
    if progress_callback:
        progress_callback(30)

    # Step 3: Comparisons (30% to 90%)
    mse_scores, ssim_scores, temp_scores, hist_scores, hog_scores, nmi_scores = [], [], [], [], [], []
    total_comparisons = len(segmented_signatures_16x16)

    for i in range(total_comparisons):
        log_message(f"Processing signature {i + 1}...")
        comparison_results = cm.compare_signature(
            original_segments_list=segmented_signatures_16x16,
            test_segments=segmented_signatures_16x16[i],
            weights=None,
            skip_identical=True
        )

        mse_scores.append(np.mean([result['mse_avg'] for result in comparison_results]))
        ssim_scores.append(np.mean([result['ssim_avg'] for result in comparison_results]))
        temp_scores.append(np.mean([result['template_avg'] for result in comparison_results]))
        hist_scores.append(np.mean([result['histogram_avg'] for result in comparison_results]))
        hog_scores.append(np.mean([result['hog_avg'] for result in comparison_results]))
        nmi_scores.append(np.mean([result['nmi_avg'] for result in comparison_results]))

        # Update progress incrementally
        progress_callback(30 + (i + 1) * 40 // total_comparisons)

    # Step 5: Compute and save scores (90% to 100%)
    log_message("Computing scores...")
    scores = {
        "mse": float(np.mean(mse_scores)),
        "ssim": float(np.mean(ssim_scores)),
        "template": float(np.mean(temp_scores)),
        "histogram": float(np.mean(hist_scores)),
        "hog": float(np.mean(hog_scores)),
        "nmi": float(np.mean(nmi_scores)),
    }

    with open('Flask app/scores.json', 'w') as json_file:
        json.dump(scores, json_file)

    log_message("Scores saved successfully.")
    if progress_callback:
        progress_callback(100)
