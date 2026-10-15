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
"""

import rules
import cv2
import numpy as np
#: Import the original rules and append your new one.
RULES = rules.RULES + ("rule4_edges",)

# Helper functions 
@lru_cache(max_size=512)
def _orb_features(path):
    gray_img = rules._gray(path)

    if gray_img == None:
        return None, None

    orb = cv2.ORB_CREATE(nfeatures=1500)
    return orb.detectAndCompute(gray, None)



def rule4_orb(target, input_path):
    pass

def rule4_sift(target, input_path):
    pass

def rule4_edges(target, input_path):
    """TODO: replace with your Phase 2 rule that fixes the V1 weakness.

    Starter returns a no-op so the pipeline still runs before you implement it.
    Parity with rules.py rule dict: rule/name/fired/score/out_of/note/metric.
    """
    return {
        "rule": 4,
        "name": "Edges",
        "fired": False,
        "score": 0,
        "out_of": 10,
        "note": "Not implemented",
        "metric": 0.0,
    }
