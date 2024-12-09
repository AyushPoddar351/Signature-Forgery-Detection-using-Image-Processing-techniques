import numpy as np
import cv2

from skimage.metrics import structural_similarity as compare_ssim

# Function to calculate MSE between two segments
def calculate_mse(segment1, segment2):
        return np.mean((segment1.astype("float") - segment2.astype("float")) ** 2)

# Function to calculate SSIM between two segments
def calculate_ssim(segment1, segment2):
        smallest_dim = min(segment1.shape[0], segment1.shape[1])
        win_size = min(7, smallest_dim) if smallest_dim >= 7 else smallest_dim
        if win_size % 2 == 0:  # Ensure win_size is odd
            win_size -= 1
        score, _ = compare_ssim(segment1, segment2, full=True, win_size=win_size)
        return score

# Template Matching Function
def calculate_template_match(segment1, segment2):
        if segment1.shape != segment2.shape:
            segment2 = cv2.resize(segment2, (segment1.shape[1], segment1.shape[0]))
        match_result = cv2.matchTemplate(segment1, segment2, cv2.TM_CCOEFF_NORMED)
        return np.max(match_result)

# Histogram Comparison Function
def calculate_histogram_similarity(segment1, segment2):
        hist1 = cv2.calcHist([segment1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([segment2], [0], None, [256], [0, 256])
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return score

from skimage.feature import hog

# Function to calculate HOG similarity between two segments
def calculate_hog_similarity(segment1, segment2):
        # Compute HOG features for both segments
        hog1, _ = hog(segment1, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        hog2, _ = hog(segment2, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)

        # Compute the norms of the HOG feature vectors
        norm1 = np.linalg.norm(hog1)
        norm2 = np.linalg.norm(hog2)

        # Check if any of the norms is zero and handle it
        if norm1 == 0 or norm2 == 0:
            return 0  # Return 0 similarity if one of the vectors is a zero vector

        # Compute the cosine similarity between the two HOG feature vectors
        score = np.dot(hog1, hog2) / (norm1 * norm2)
        return score

from sklearn.metrics import normalized_mutual_info_score

# Normalized Mutual Information Function
def calculate_nmi(segment1, segment2):
        hist1, _ = np.histogram(segment1.ravel(), bins=256, range=(0, 256))
        hist2, _ = np.histogram(segment2.ravel(), bins=256, range=(0, 256))
        score = normalized_mutual_info_score(hist1, hist2)
        return score

def compare_signature(original_segments_list, test_segments, weights=None, skip_identical=True):
        results = []
        for idx, original_segments in enumerate(original_segments_list):
            if skip_identical and original_segments is test_segments:
                continue
            
            scores = {"mse": [], "ssim": [], "template": [], "histogram": [], "hog": [], "dssim": [], "nmi": []}
            for i in range(len(original_segments)):
                if np.array_equal(original_segments[i], test_segments[i]):
                    continue

                mse_score = calculate_mse(original_segments[i], test_segments[i])
                scores["mse"].append(1 / (1 + mse_score))  # Normalize MSE

                ssim_score = calculate_ssim(original_segments[i], test_segments[i])
                scores["ssim"].append(ssim_score)

                template_score = calculate_template_match(original_segments[i], test_segments[i])
                scores["template"].append(template_score)

                hist_score = calculate_histogram_similarity(original_segments[i], test_segments[i])
                scores["histogram"].append(hist_score)

                hog_score = calculate_hog_similarity(original_segments[i], test_segments[i])
                scores["hog"].append(hog_score)

                nmi_score = calculate_nmi(original_segments[i], test_segments[i])
                scores["nmi"].append(nmi_score)

            mse_avg = np.average(scores["mse"], weights=weights)
            ssim_avg = np.average(scores["ssim"], weights=weights)
            template_avg = np.average(scores["template"], weights=weights)
            histogram_avg = np.average(scores["histogram"], weights=weights)
            hog_avg = np.average(scores["hog"], weights=weights)
            nmi_avg = np.average(scores["nmi"], weights=weights)

            results.append({
                "mse_avg": mse_avg,
                "ssim_avg": ssim_avg,
                "template_avg": template_avg,
                "histogram_avg": histogram_avg,
                "hog_avg": hog_avg,
                "nmi_avg": nmi_avg,
            })
        
        return results