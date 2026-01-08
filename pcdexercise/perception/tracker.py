import numpy as np
from scipy.optimize import linear_sum_assignment

class Track:
    def __init__(self, id, pos):
        self.id = id
        self.pos = pos
        self.vel = np.zeros(3)
        self.acc = np.zeros(3)
        self.jerk = np.zeros(3)
        self.traj = []

class Tracker:
    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def update(self, detections, dt=0.1):
        if not self.tracks:
            for d in detections:
                self.tracks.append(Track(self.next_id, d))
                self.next_id += 1
            return

        cost = np.linalg.norm(
            np.array([[np.linalg.norm(t.pos - d) for d in detections] for t in self.tracks]),
            axis=2
        )
        row,col = linear_sum_assignment(cost)

        assigned=set()
        used=set()

        for r,c in zip(row,col):
            t = self.tracks[r]
            d = detections[c]
            prev = t.pos.copy()
            t.pos = d
            t.vel = (t.pos-prev)/dt
            t.acc = (t.vel-t.vel)/dt  # simple approx
            t.jerk = (t.acc-t.acc)/dt
            t.traj.append(t.pos)
            used.add(c)
            assigned.add(r)

        for i,d in enumerate(detections):
            if i not in used:
                self.tracks.append(Track(self.next_id, d))
                self.next_id+=1

        self.tracks=[t for i,t in enumerate(self.tracks) if i in assigned or True]
