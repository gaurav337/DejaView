import cv2
import numpy as np
import matplotlib.pyplot as plt

TARGET_SIZE = (500, 500)

ORB_FEATURES = 500      
MATCH_THRESHOLD = 50    

WEIGHTS = {
    'structure': 0.42,  
    'spatial':   0.43,  
    'color':     0.15  
}

print(" Configuration Loaded.")
print(f"   Weights: {WEIGHTS}")

def preprocess_image(img):
    if img is None:
        return None

    resized = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(resized, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    return blurred, gray, hsv

def get_histogram_score(img_a, img_b, channels, bins, ranges):
    hist_a = cv2.calcHist([img_a], channels, None, bins, ranges)
    hist_b = cv2.calcHist([img_b], channels, None, bins, ranges)
    cv2.normalize(hist_a, hist_a, 1, 0, cv2.NORM_L1)
    cv2.normalize(hist_b, hist_b, 1, 0, cv2.NORM_L1)
    return cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_INTERSECT)

def get_orb_score(gray_a, gray_b):
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    kp1, des1 = orb.detectAndCompute(gray_a, None)
    kp2, des2 = orb.detectAndCompute(gray_b, None)
    
    raw_matches = 0
    if des1 is not None and des2 is not None:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        raw_matches = len(matches)
        
    return min(1.0, raw_matches / MATCH_THRESHOLD), raw_matches

def compare_image_data(data_a, data_b):
   
    _, gray_a, hsv_a = data_a
    _, gray_b, hsv_b = data_b
    
    s_struct = get_histogram_score(gray_a, gray_b, [0], [50], [0, 256])
    s_color = get_histogram_score(hsv_a, hsv_b, [0, 1], [30, 10], [0, 180, 0, 256])
    s_spatial, raw_count = get_orb_score(gray_a, gray_b)
    
    final_score = (
        (s_struct  * WEIGHTS['structure']) +
        (s_spatial * WEIGHTS['spatial']) +
        (s_color   * WEIGHTS['color'])
    )
    
    return {
        'total': final_score,
        'details': {
            'structure': s_struct,
            'spatial': s_spatial,
            'color': s_color,
            'raw_matches': raw_count
        }
    }

path_a = '3.png'
path_b = '5.png'

img_a_raw = cv2.imread(path_a)
img_b_raw = cv2.imread(path_b)

if img_a_raw is None or img_b_raw is None:
    print("Error: Could not load images.")
else:
    data_a = preprocess_image(img_a_raw)
    
    data_b_normal = preprocess_image(img_b_raw)
    result_normal = compare_image_data(data_a, data_b_normal)
    
    img_b_flipped = cv2.flip(img_b_raw, 1)
    data_b_flipped = preprocess_image(img_b_flipped)
    result_flipped = compare_image_data(data_a, data_b_flipped)
    
    if result_flipped['total'] > result_normal['total']:
        final_result = result_flipped
        match_type = "Mirrored / Flipped"
        best_b_img = img_b_flipped
    else:
        final_result = result_normal
        match_type = "Standard"
        best_b_img = img_b_raw

    score = final_result['total']
    d = final_result['details']
    
    print(f"🔎 Status: {match_type}")
    print(f"---------------------------")
    print(f"Structure: {d['structure']:.2f}")
    print(f"Spatial:   {d['spatial']:.2f} (Matches: {d['raw_matches']})")
    print(f"Color:     {d['color']:.2f}")
    print(f"---------------------------")
    print(f"FINAL SCORE: {score:.4f} ({score*100:.1f}%)")
    
    if score > 0.80: print("Duplicate / Edit")
    elif score > 0.60: print("Similar")
    else: print("Different")

    # Show images
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1); plt.imshow(cv2.cvtColor(img_a_raw, cv2.COLOR_BGR2RGB)); plt.title("A")
    plt.subplot(1,2,2); plt.imshow(cv2.cvtColor(best_b_img, cv2.COLOR_BGR2RGB)); plt.title(f"B ({match_type})")
    plt.show()    