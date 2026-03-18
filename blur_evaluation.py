import cv2
import os
import re
import numpy as np

def extract_prediction_row(image_path):
    """
    Crops the bottom half of your comparison plots to isolate 
    the predicted frames from the target frames and labels.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    h, w, _ = img.shape
    # Since your plot has 2 rows, the bottom half (h//2 to h) 
    # contains the predictions.
    pred_row = img[h // 2 : h, :]
    return cv2.cvtColor(pred_row, cv2.COLOR_BGR2GRAY)

def get_laplacian_var(gray_img):
    """Calculates the variance of the Laplacian."""
    return cv2.Laplacian(gray_img, cv2.CV_64F).var()

def analyze_epoch_samples(sample_dir):
    if not os.path.exists(sample_dir):
        print(f"Directory {sample_dir} not found.")
        return

    # Natural sort to keep epochs in order (epoch1, epoch2... epoch10)
    files = [f for f in os.listdir(sample_dir) if f.endswith('.png')]
    files.sort(key=lambda var: [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', var)])

    epoch_scores = []

    print(f"{'File Name':<25} | {'Blur Score':<10}")
    print("-" * 40)

    for filename in files:
        path = os.path.join(sample_dir, filename)
        pred_segment = extract_prediction_row(path)
        
        if pred_segment is not None:
            score = get_laplacian_var(pred_segment)
            epoch_scores.append(score)
            print(f"{filename:<25} | {score:.2f}")

    if epoch_scores:
        avg_score = sum(epoch_scores) / len(epoch_scores)
        print("-" * 40)
        print(f"Average Blur Score across {len(epoch_scores)} epochs: {avg_score:.2f}")
        print("Note: Higher scores = Sharper edges.")
    else:
        print("No valid images found.")

if __name__ == "__main__":
    analyze_epoch_samples("samples/preds")