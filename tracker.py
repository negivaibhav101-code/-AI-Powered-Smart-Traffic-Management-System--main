from collections import deque

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
        used = set()
        for tid, tr in list(self.tracks.items()):
            best_j, best_d = -1, 1e9
            for j, (x,y) in enumerate(detections):
                if j in used: continue
                d = (tr.x - x)**2 + (tr.y - y)**2
                if d < best_d:
                    best_d, best_j = d, j
            if best_j >= 0 and best_d <= self.dist_thresh**2:
                x,y = detections[best_j]
                used.add(best_j)
                tr.vx = 0.6*tr.vx + 0.4*(x - tr.x)
                tr.vy = 0.6*tr.vy + 0.4*(y - tr.y)
                tr.x, tr.y = x, y
                tr.trail.append((x,y))
                tr.miss = 0
                tr.age += 1
            else:
                tr.miss += 1

        for j, (x,y) in enumerate(detections):
            if j in used: continue
            tid = self.next_id; self.next_id += 1
            self.tracks[tid] = Track(tid, x, y)

        for tid in list(self.tracks.keys()):
            if self.tracks[tid].miss > self.max_miss:
                del self.tracks[tid]

        return self.tracks