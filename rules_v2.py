"""EAS 510 - Project 1 - Phase 2: the V2 rule set.

Rules 1-3 are kept exactly as in `rules.py`. Your job is to ADD Rule 4 that
targets the systematic weakness you diagnosed from results_v1_hard.txt.

Constraint: the rules' `out_of` weights must still sum to 100 so the Final
Score stays on a /100 scale (the format validator enforces "Final Score:
<s>/100" equals the sum of the block's rule scores).

Example split that keeps the original spirit: Rules 1-3 -> 25/25/40 and
Rule 4 -> 10 (or trim the losers more aggressively).

RULE 1 WEAKNESSES: 
    - metadata isn't strong evidence of image-matching.
    - cropping and resizing change dimentions and area ratios or compression.
    - dimentions, aspect ratio , files formats and JPEG-PNG only shows a format conversion but not identifying 
      which original the image came from. 
    - Ofc lastly problem with rotations, image augmentations that are still match if human were to observe it. 

    SO my current thinking as i have taken long way before Computer Vision Course. I am looking into implmenting ORB-KEYPOINTS, SIFT W/RANSAC i know those two are used in image stitching to build panaroma however, i believe they can be used in image matching. 
    - Also another algorithm can be used which clustering but it can count as an ML model mostly it is used in Face detections algorithms. 

    - We will experement first with the first two. 
Resources: 
    - RANSAC: https://medium.com/@717821i208/ransac-3187c732ac5d +cv2: https://docs.opencv.org/3.4.20/d1/de0/tutorial_py_feature_homography.html 
    - The idea of KNN came from the above opencv documentation. 
    - ORB: https://medium.com/analytics-vidhya/understanding-keypoints-and-orb-algorithm-f260e6fb640c
    - ORB: https://medium.com/thedeephub/detecting-and-tracking-objects-with-orb-using-opencv-d228f4c9054e
    - GEEKFORGEEKS FOR RANSAC IMPLEMENTATION: https://www.geeksforgeeks.org/python/step-by-step-guide-to-using-ransac-in-opencv-using-python/ 
"""

import rules
import cv2
import numpy as np
from functools import lru_cache
#: Import the original rules and append your new one.
RULES = rules.RULES + ("rule4_edges",)

# Helper functions 
@lru_cache(maxsize=512)
def _orb_features(path):
    gray_img = rules._gray(path)

    if gray_img is None:
        return None, None

    orb = cv2.ORB_create(nfeatures=1500)
    return orb.detectAndCompute(gray_img, None)

def ransac(source_keypoints, suspect_keypoints, matches):
    if len(matches) < 4:
        return 0, 0.0

    source_points = np.float32([source_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    suspect_points = np.float32([suspect_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    _, mask = cv2.findHomography(suspect_points, source_points, cv2.RANSAC, 5.0)

    if mask is None:
        return 0, 0.0

    inliers = int(mask.ravel().sum())
    inliers_ratio = inliers / len(matches)

    return inliers, inliers_ratio

def rule4_orb(target, input_path):
    out = {"rule": 4, "name": "Edges", "fired": False, "score": 0,
           "out_of": 10, "note": "ransac inlier 0.0/0.0", "metric": 0.0}

    try:
        source_keypoints, source_description = _orb_features(target["path"])
        suspect_keypoints, suspect_description = _orb_features(input_path)

        if (source_keypoints is None 
            or source_description is None 
            or suspect_keypoints is None 
            or suspect_description is None 
            or len(source_description) < 2):
            return out

        # So next to find distance between source_keypoints and suspect_keypoints. Using humming distance &knn 
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        raw_knn_matches = bf.knnMatch(suspect_description, source_description, k=2) #try k = 2 we can experiment with this. 

        good_matches = []

        for match_pair in raw_knn_matches:
            if len(match_pair) == 2:
                first, second = match_pair 
                if first.distance < 0.75 * second.distance:
                    good_matches.append(first)

        inliers, inliers_ratio = ransac(source_keypoints, suspect_keypoints, good_matches)
        
        out["note"] = f"Ransac inlier {inliers}/{len(good_matches)}"

        metric = min(1.0, inliers / 10.0) * inliers_ratio
        out["metric"] = round(metric, 3)

        if inliers >= 6 and inliers_ratio >= 0.40:
            out["fired"] = True
            out["score"] = int(round(out["out_of"] * metric))
    except Exception as e:
        out["note"] = f"ORB ERROR : {type(e).__name__}"

    return out

def rule4_edges(target, input_path):
    """TODO: replace with your Phase 2 rule that fixes the V1 weakness.

    Starter returns a no-op so the pipeline still runs before you implement it.
    Parity with rules.py rule dict: rule/name/fired/score/out_of/note/metric.
    """
    return rule4_orb(target, input_path)


