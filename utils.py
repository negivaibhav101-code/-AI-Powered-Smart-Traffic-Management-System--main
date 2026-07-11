


import os
import json
import cv2
import numpy as np

def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(np.array(poly, dtype=np.int32), pt, False) >= 0

def draw_filled_poly(img, poly, color, alpha=0.18):
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array(poly, dtype=np.int32)], color, lineType=cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def put_text(img, text, org, scale=0.6, color=(255,255,255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def clamp(v, lo, hi): 
    return max(lo, min(hi, v))

def make_auto_rois(w, h):
    cx, cy = w//2, h//2
    north = [(0,0), (w,0), (cx,cy)]
    south = [(0,h), (w,h), (cx,cy)]
    west  = [(0,0), (0,h), (cx,cy)]
    east  = [(w,0), (w,h), (cx,cy)]
    cw, ch = int(0.26*w), int(0.26*h)  
    center = [(cx-cw//2, cy-ch//2), (cx+cw//2, cy-ch//2), (cx+cw//2, cy+ch//2), (cx-cw//2, cy+ch//2)]
    return {"N": north, "S": south, "E": east, "W": west, "CENTER": center}

def load_rois(roi_path, w, h):
    if not roi_path or not os.path.exists(roi_path):
        return make_auto_rois(w, h)
    with open(roi_path, "r") as f:
        data = json.load(f)
    for k in data:
        data[k] = [tuple(p) for p in data[k]]
    if "CENTER" not in data:
        cx, cy = w//2, h//2
        cw, ch = int(0.26*w), int(0.26*h)
        data["CENTER"] = [(cx-cw//2, cy-ch//2), (cx+cw//2, cy-ch//2),
                          (cx+cw//2, cy+ch//2), (cx-cw//2, cy+ch//2)]
    return data