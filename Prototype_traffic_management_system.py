import argparse, time, os, json, math
from collections import deque
import numpy as np
import cv2
from ultralytics import YOLO

# -----------------------------
# Utility: polygons & drawing
# -----------------------------
def point_in_poly(pt, poly):
    return cv2.pointPolygonTest(np.array(poly, dtype=np.int32), pt, False) >= 0

def draw_filled_poly(img, poly, color, alpha=0.18):
    overlay = img.copy()
    cv2.fillPoly(overlay, [np.array(poly, dtype=np.int32)], color, lineType=cv2.LINE_AA)
    return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

def put_text(img, text, org, scale=0.6, color=(255,255,255), thick=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

def clamp(v, lo, hi): return max(lo, min(hi, v))

# -----------------------------
# Auto-ROIs: 4 triangular approaches + center box
# Works for most 4-legs CCTV/drone clips (like MNn9qKG2UFI)
# -----------------------------
def make_auto_rois(w, h):
    cx, cy = w//2, h//2
    north = [(0,0), (w,0), (cx,cy)]
    south = [(0,h), (w,h), (cx,cy)]
    west  = [(0,0), (0,h), (cx,cy)]
    east  = [(w,0), (w,h), (cx,cy)]
    cw, ch = int(0.26*w), int(0.26*h)  # center box for spillback detection
    center = [(cx-cw//2, cy-ch//2), (cx+cw//2, cy-ch//2), (cx+cw//2, cy+ch//2), (cx-cw//2, cy+ch//2)]
    return {"N": north, "S": south, "E": east, "W": west, "CENTER": center}

def load_rois(roi_path, w, h):
    if not roi_path or not os.path.exists(roi_path):
        return make_auto_rois(w, h)
    with open(roi_path, "r") as f:
        data = json.load(f)
    # ensure tuples
    for k in data:
        data[k] = [tuple(p) for p in data[k]]
    if "CENTER" not in data:
        cx, cy = w//2, h//2
        cw, ch = int(0.26*w), int(0.26*h)
        data["CENTER"] = [(cx-cw//2, cy-ch//2), (cx+cw//2, cy-ch//2),
                          (cx+cw//2, cy+ch//2), (cx-cw//2, cy+ch//2)]
    return data

# -----------------------------
# Simple multi-object tracker (ID + motion trail)
# Greedy nearest-neighbor match (good enough for a show demo)
# -----------------------------
class Track:
    __slots__ = ("id","x","y","vx","vy","trail","age","miss")
    def __init__(self, id, x, y):
        self.id = id
        self.x, self.y = x, y
        self.vx, self.vy = 0.0, 0.0
        self.trail = deque(maxlen=32)
        self.trail.append((x,y))
        self.age = 0
        self.miss = 0

class Tracker:
    def __init__(self, dist_thresh=48, max_miss=10):
        self.tracks = {}
        self.next_id = 1
        self.dist_thresh = dist_thresh
        self.max_miss = max_miss

    def update(self, detections):
        # detections: list of (x,y)
        used = set()
        # assign by nearest neighbor
        for tid, tr in list(self.tracks.items()):
            # find nearest detection
            best_j, best_d = -1, 1e9
            for j, (x,y) in enumerate(detections):
                if j in used: continue
                d = (tr.x - x)**2 + (tr.y - y)**2
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0 and best_d <= self.dist_thresh**2:
                x,y = detections[best_j]
                used.add(best_j)
                # velocity
                tr.vx = 0.6*tr.vx + 0.4*(x - tr.x)
                tr.vy = 0.6*tr.vy + 0.4*(y - tr.y)
                tr.x, tr.y = x, y
                tr.trail.append((x,y))
                tr.miss = 0
                tr.age += 1
            else:
                tr.miss += 1

        # spawn new tracks for unmatched detections
        for j, (x,y) in enumerate(detections):
            if j in used: continue
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = Track(tid, x, y)

        # cull old
        for tid in list(self.tracks.keys()):
            if self.tracks[tid].miss > self.max_miss:
                del self.tracks[tid]

        return self.tracks

# -----------------------------
# Max-Pressure Controller with spillback, min/max green, clearance, optional emergency
# Phase 0 = NS green, Phase 1 = EW green
# -----------------------------
class MaxPressureController:
    def __init__(self, epoch=2.0, min_green=4.0, max_green=20.0, clearance=2.0, alpha=0.3,
                 spill_limit=14, emergency_at=0.0, emergency_for=0.0, emergency_dir="EW"):
        self.EPOCH = epoch
        self.MIN_GREEN = min_green
        self.MAX_GREEN = max_green
        self.CLEARANCE = clearance
        self.ALPHA = alpha
        self.SPILL_LIM = spill_limit
        self.emergency_at = emergency_at
        self.emergency_for = emergency_for
        self.emergency_dir = emergency_dir.upper()  # "NS" or "EW"

        self.phase = 0
        self.t_in_phase = 0.0
        self.clear_left = 0.0
        self.since_decide = 0.0

    def _emg_active(self, t):
        if self.emergency_at <= 0 or self.emergency_for <= 0: return False
        return self.emergency_at <= t < (self.emergency_at + self.emergency_for)

    def _pressures(self, qN, qS, qE, qW, center_load):
        q_ns = qN + qS
        q_ew = qE + qW
        dn_ns = dn_ew = 0
        if center_load >= self.SPILL_LIM:
            dn_ns = center_load
            dn_ew = center_load
        pNS = q_ns - self.ALPHA*dn_ns
        pEW = q_ew - self.ALPHA*dn_ew
        return pNS, pEW

    def choose(self, dt, tnow, qN, qS, qE, qW, center_load):
        self.t_in_phase += dt
        self.since_decide += dt

        # clearance running
        if self.clear_left > 0:
            self.clear_left -= dt
            return self.phase, True, self._emg_active(tnow)

        # emergency
        if self._emg_active(tnow):
            target = 1 if self.emergency_dir == "EW" else 0
            if self.phase != target and self.t_in_phase >= self.MIN_GREEN:
                self.clear_left = self.CLEARANCE
            return self.phase, False, True

        # decision epoch or forced flip at max green
        want_decide = self.since_decide >= self.EPOCH
        force_flip = self.t_in_phase >= self.MAX_GREEN

        if want_decide or force_flip:
            pNS, pEW = self._pressures(qN,qS,qE,qW,center_load)
            best = 0 if pNS >= pEW else 1

            if force_flip and self.clear_left <= 0:
                if self.t_in_phase >= self.MIN_GREEN:
                    self.clear_left = self.CLEARANCE
                    self.since_decide = 0.0
                    return self.phase, True, False

            if best != self.phase:
                if self.t_in_phase >= self.MIN_GREEN:
                    self.clear_left = self.CLEARANCE
                    self.since_decide = 0.0
                    return self.phase, True, False
                # else hold min-green

            self.since_decide = 0.0

        return self.phase, False, False

    def apply_after_clearance(self):
        if self.clear_left <= 0:
            self.phase = 1 - self.phase
            self.t_in_phase = 0.0

# -----------------------------
# Dashboard drawing (right panel)
# -----------------------------
def draw_dashboard(frame, panel_w, counts, pressures, phase, timers, spark_hist):
    """
    counts: (qN,qS,qE,qW,center)
    pressures: (pNS,pEW)
    phase: 0=NS green, 1=EW green
    timers: (t_in_phase, clear_left, epoch, min_g, max_g)
    spark_hist: list of recent total queues
    """
    h, w = frame.shape[:2]
    x0 = w - panel_w
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)

    # header
    put_text(panel, "SMART SIGNAL CONTROLLER", (18, 40), 0.7, (255,255,255), 2)

    # phase badge
    badge_col = (0,200,0) if phase==0 else (0,120,255)
    cv2.rectangle(panel, (16, 60), (panel_w-16, 110), badge_col, -1, cv2.LINE_AA)
    ph_text = f"PHASE: {'NS' if phase==0 else 'EW'}"
    put_text(panel, ph_text, (26, 95), 0.9, (255,255,255), 2)

    # counts
    qN,qS,qE,qW,c = counts
    put_text(panel, f"N:{qN:02d}  S:{qS:02d}  E:{qE:02d}  W:{qW:02d}  C:{c:02d}", (18, 140), 0.65)

    # pressure bars
    pNS, pEW = pressures
    maxp = max(1.0, abs(pNS), abs(pEW))
    bar_len_NS = int((panel_w-60) * (pNS/maxp))
    bar_len_EW = int((panel_w-60) * (pEW/maxp))
    y_ns = 180; y_ew = 210
    cv2.rectangle(panel, (30, y_ns-12), (30+bar_len_NS, y_ns+12), (0,220,0), -1, cv2.LINE_AA)
    cv2.rectangle(panel, (30, y_ew-12), (30+bar_len_EW, y_ew+12), (0,160,255), -1, cv2.LINE_AA)
    put_text(panel, f"P(NS)={pNS:.1f}", (32, y_ns-18), 0.55)
    put_text(panel, f"P(EW)={pEW:.1f}", (32, y_ew-18), 0.55)

    # timers
    t_in, clear_left, epoch, min_g, max_g = timers
    put_text(panel, f"t_in_phase: {t_in:4.1f}s", (18, 250), 0.55)
    put_text(panel, f"clearance:  {max(0.0,clear_left):4.1f}s", (18, 275), 0.55)
    put_text(panel, f"epoch/min/max: {epoch:.0f}/{min_g:.0f}/{max_g:.0f}s", (18, 300), 0.55)

    # sparkline of total queue
    chart_h = 90
    chart_y0 = 330
    cv2.rectangle(panel, (18, chart_y0), (panel_w-18, chart_y0+chart_h), (70,70,70), 1, cv2.LINE_AA)
    if len(spark_hist) >= 2:
        vals = np.array(spark_hist[-80:], dtype=float)
        mn, mx = np.min(vals), np.max(vals)
        rng = (mx - mn) if (mx > mn) else 1.0
        xs = np.linspace(22, panel_w-22, len(vals)).astype(int)
        ys = (chart_y0 + chart_h - 4 - (vals - mn) * (chart_h - 8)/rng).astype(int)
        for i in range(1, len(xs)):
            cv2.line(panel, (xs[i-1], ys[i-1]), (xs[i], ys[i]), (200,200,200), 2, cv2.LINE_AA)
        put_text(panel, f"Total Q: {int(vals[-1])}", (22, chart_y0+chart_h+22), 0.6)

    # legend
    put_text(panel, "Decision = argmax Pressure", (18, h-70), 0.55, (220,220,220), 1)
    put_text(panel, "Spillback guard via CENTER load", (18, h-46), 0.55, (220,220,220), 1)

    frame[:, x0:] = panel
    return frame

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Decision-making demo on real video (YOLO + Max-Pressure)")
    ap.add_argument("--video", default="Surveillance Camera Footage.mp4")
    ap.add_argument("--out", default="out_1decision.mp4")
    ap.add_argument("--roi", default="", help="Optional ROI JSON with N,S,E,W,(CENTER)")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--device", default=None, help="mps/cpu/cuda")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--epoch", type=float, default=2.0)
    ap.add_argument("--min_green", type=float, default=4.0)
    ap.add_argument("--max_green", type=float, default=18.0)
    ap.add_argument("--clearance", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--spill_limit", type=int, default=14)
    ap.add_argument("--emergency_at", type=float, default=0.0)   # set >0 to demo preemption
    ap.add_argument("--emergency_for", type=float, default=0.0)
    ap.add_argument("--emergency_dir", default="EW", choices=["EW","NS"])
    ap.add_argument("--max_frames", type=int, default=0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    # Add a dashboard panel on the right
    PANEL_W = int(0.32 * W)
    OUT_W = W + PANEL_W
    OUT_H = H

    out = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (OUT_W, OUT_H))

    rois = load_rois(args.roi, W, H)

    # YOLO
    model = YOLO(args.model)
    if args.device:
        model.to(args.device)

    # controller + tracker
    ctrl = MaxPressureController(epoch=args.epoch, min_green=args.min_green, max_green=args.max_green,
                                 clearance=args.clearance, alpha=args.alpha, spill_limit=args.spill_limit,
                                 emergency_at=args.emergency_at, emergency_for=args.emergency_for,
                                 emergency_dir=args.emergency_dir)
    tracker = Tracker(dist_thresh=48, max_miss=10)

    # smoothing and history
    win = max(1, int(0.4 * FPS))
    qN_hist, qS_hist, qE_hist, qW_hist, qC_hist = deque(maxlen=win), deque(maxlen=win), deque(maxlen=win), deque(maxlen=win), deque(maxlen=win)
    totalQ_hist = []

    frame_idx = 0
    VEH_LABELS = {"car","truck","bus","motorcycle","motorbike","bicycle"}

    while True:
        ok, frame0 = cap.read()
        if not ok:
            break
        frame_idx += 1
        if args.max_frames and frame_idx > args.max_frames:
            break

        # base canvas (original + darken a bit for overlays)
        base = frame0.copy()
        base = (base * 0.94).astype(np.uint8)

        # draw region overlays (soft)
        for key,col in [("N",(60,200,60)),("S",(60,200,60)),("E",(60,120,255)),("W",(60,120,255)),("CENTER",(0,210,210))]:
            base = draw_filled_poly(base, rois[key], col, 0.10 if key!="CENTER" else 0.12)
            cv2.polylines(base, [np.array(rois[key], np.int32)], True, (160,160,160), 1, cv2.LINE_AA)
            put_text(base, key, tuple(np.mean(np.array(rois[key]), axis=0).astype(int)), 0.5, (240,240,240), 1)

        # detections
        res = model(base, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        det_pts = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            clsi  = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy()
            names = res.names
            for (x1,y1,x2,y2), ci, cf in zip(boxes, clsi, confs):
                lbl = names.get(ci, str(ci)) if isinstance(names, dict) else names[ci]
                if lbl in VEH_LABELS:
                    cx = int((x1 + x2)/2)
                    cy = int((y1 + y2)/2)
                    det_pts.append((cx,cy))

        # update tracks & draw trails (no boxes)
        tracks = tracker.update(det_pts)
        for tid, tr in tracks.items():
            # color by rough heading (vx,vy)
            spd = math.hypot(tr.vx, tr.vy)
            if abs(tr.vx) > abs(tr.vy):  # E-W
                col = (0,140,255)  # orange-blue
            else:
                col = (0,200,0)    # green
            # trail
            for i in range(1, len(tr.trail)):
                cv2.line(base, tr.trail[i-1], tr.trail[i], col, 2, cv2.LINE_AA)
            # centroid
            cv2.circle(base, (int(tr.x), int(tr.y)), 3, (255,255,255), -1, cv2.LINE_AA)

        # counts per ROI (by centroid)
        qN=qS=qE=qW=qC=0
        for (x,y) in det_pts:
            if point_in_poly((x,y), rois["CENTER"]): qC += 1
            if point_in_poly((x,y), rois["N"]): qN += 1
            if point_in_poly((x,y), rois["S"]): qS += 1
            if point_in_poly((x,y), rois["E"]): qE += 1
            if point_in_poly((x,y), rois["W"]): qW += 1

        # smooth a bit
        qN_hist.append(qN); qS_hist.append(qS); qE_hist.append(qE); qW_hist.append(qW); qC_hist.append(qC)
        qN_s = int(np.mean(qN_hist)); qS_s = int(np.mean(qS_hist))
        qE_s = int(np.mean(qE_hist)); qW_s = int(np.mean(qW_hist)); qC_s = int(np.mean(qC_hist))

        # decision step
        dt = 1.0 / float(FPS)
        tnow = frame_idx / float(FPS)
        phase_before = ctrl.phase
        phase, go_clear, emg = ctrl.choose(dt, tnow, qN_s, qS_s, qE_s, qW_s, qC_s)
        if go_clear and ctrl.clear_left <= 0:
            ctrl.apply_after_clearance()

        # pretty phase bars on main view
        bar_h = int(0.03 * H)
        if ctrl.clear_left > 0:  # clearance
            color_ns = color_ew = (0, 220, 220)
        else:
            if ctrl.phase == 0:   # NS green, EW red
                color_ns = (0, 190, 0)
                color_ew = (0, 0, 200)
            else:
                color_ns = (0, 0, 200)
                color_ew = (0, 190, 0)
        cv2.rectangle(base, (0,0), (W, bar_h), color_ns, -1, cv2.LINE_AA)       # top (N)
        cv2.rectangle(base, (0,H-bar_h), (W,H), color_ns, -1, cv2.LINE_AA)      # bottom (S)
        cv2.rectangle(base, (0,0), (int(0.03*W), H), color_ew, -1, cv2.LINE_AA) # left (W)
        cv2.rectangle(base, (W-int(0.03*W),0), (W,H), color_ew, -1, cv2.LINE_AA)# right (E)

        # dashboard
        pNS = (qN_s + qS_s) - args.alpha * (qC_s if qC_s >= args.spill_limit else 0)
        pEW = (qE_s + qW_s) - args.alpha * (qC_s if qC_s >= args.spill_limit else 0)
        totalQ_hist.append(qN_s + qS_s + qE_s + qW_s)

        composed = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
        composed[:, :W] = base
        composed = draw_dashboard(
            composed, PANEL_W,
            counts=(qN_s,qS_s,qE_s,qW_s,qC_s),
            pressures=(pNS,pEW),
            phase=ctrl.phase,
            timers=(ctrl.t_in_phase, ctrl.clear_left, args.epoch, args.min_green, args.max_green),
            spark_hist=totalQ_hist
        )

        # footer caption
        put_text(composed, "Decision = argmax{ Pressure(NS), Pressure(EW) } with spillback guard & safety timings",
                 (18, OUT_H-12), 0.6, (255,255,255), 2)

        out.write(composed)

    cap.release()
    out.release()
    print(f"[OK] Saved: {args.out}")

if __name__ == "__main__":
    main()
