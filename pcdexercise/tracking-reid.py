"""
tracking.py
-----------
Robust multi-object human tracking in 3D point clouds.

- Detection: DBSCAN clustering + geometry validation
- Tracking: IMM Kalman Filter (CV + RW)
- Association: Hungarian + gating
- Re-ID: learned embeddings (CLIP) + spatial proximity fallback
- Output: tracking_results.json (unchanged schema)

Re-ID approach (off-the-shelf, no training):
- Project each cluster point cloud into multi-view depth images
- Extract CLIP image embeddings (open_clip)
- Aggregate multi-view embeddings => a stable descriptor for association/re-ID
"""

import os
import re
import json
import numpy as np
import open3d as o3d
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter, IMMEstimator

# Silence Open3D spam
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

# =========================
# Utility
# =========================

def extract_timestamp_ms(filename: str) -> int:
    match = re.search(r"(\d+)ms", filename)
    if not match:
        raise ValueError(f"Invalid filename timestamp: {filename}")
    return int(match.group(1))


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x) + eps
    return x / n


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    da = np.linalg.norm(a) + eps
    db = np.linalg.norm(b) + eps
    return float(1.0 - np.dot(a, b) / (da * db))


# =========================
# Learned ReID Embeddings (off-the-shelf)
# =========================

class ClipReIDEmbedder:
    """
    Produces a learned embedding for a 3D cluster by:
    1) Sampling points
    2) Creating multi-view depth projections (images)
    3) Encoding with CLIP image encoder (open_clip)
    4) Averaging and L2-normalizing

    This is "PointCLIP-style" (projection + CLIP), but kept lightweight.
    """
    def __init__(
        self,
        enabled: bool = True,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        image_size: int = 224,
        n_views: int = 6,
        points_per_cluster: int = 1024,
        depth_clip: Tuple[float, float] = (-2.5, 2.5),
    ):
        self.enabled = enabled
        self.image_size = int(image_size)
        self.n_views = int(n_views)
        self.points_per_cluster = int(points_per_cluster)
        self.depth_clip = depth_clip

        self._ok = False
        self._device = "cpu"
        self._model = None
        self._preprocess = None

        if not self.enabled:
            return

        try:
            import torch
            import open_clip
            from PIL import Image  # noqa: F401

            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            model.eval()
            model.to(self._device)

            self._torch = torch
            self._open_clip = open_clip
            self._PIL_Image = __import__("PIL.Image", fromlist=["Image"]).Image
            self._model = model
            self._preprocess = preprocess
            self._ok = True

            print(f"✅ ReID embedder enabled: CLIP {model_name} ({pretrained}) on {self._device}")

        except Exception as e:
            self._ok = False
            print(f"⚠️ ReID embedder disabled (failed to init open_clip): {e}")

    @property
    def ok(self) -> bool:
        return bool(self._ok)

    def _sample_points(self, pts: np.ndarray) -> np.ndarray:
        if pts.shape[0] == 0:
            return pts
        if pts.shape[0] >= self.points_per_cluster:
            idx = np.random.choice(pts.shape[0], self.points_per_cluster, replace=False)
            return pts[idx]
        # upsample with replacement
        idx = np.random.choice(pts.shape[0], self.points_per_cluster, replace=True)
        return pts[idx]

    def _normalize_cluster(self, pts: np.ndarray) -> np.ndarray:
        """
        Normalize to make embedding more stable:
        - center to mean
        - scale to unit radius (robust-ish)
        """
        c = pts.mean(axis=0, keepdims=True)
        pts = pts - c
        r = np.linalg.norm(pts, axis=1)
        s = np.percentile(r, 90) + 1e-6
        pts = pts / s
        return pts

    def _rotate_z(self, pts: np.ndarray, angle_rad: float) -> np.ndarray:
        ca = np.cos(angle_rad)
        sa = np.sin(angle_rad)
        R = np.array([[ca, -sa, 0.0],
                      [sa,  ca, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        return pts @ R.T

    def _render_depth_orthographic(self, pts: np.ndarray) -> np.ndarray:
        """
        Orthographic projection:
        - x,y -> image plane
        - z -> depth value
        We create a depth map where pixels take the nearest depth (max z or min z)
        For stability we use "max z" after normalization (human upright -> z separation).
        """
        H = W = self.image_size
        # map x,y from roughly [-1,1] to [0,W-1]
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]

        # clip depth for outliers
        z = np.clip(z, self.depth_clip[0], self.depth_clip[1])

        # normalize x,y to fit in view
        # (pts are already scaled; still clamp)
        x = np.clip(x, -1.2, 1.2)
        y = np.clip(y, -1.2, 1.2)

        u = ((x + 1.2) / (2.4) * (W - 1)).astype(np.int32)
        v = ((1.2 - y) / (2.4) * (H - 1)).astype(np.int32)

        depth = np.full((H, W), fill_value=-9999.0, dtype=np.float32)

        # Keep the MAX depth (closest in this normalized convention) per pixel
        # You can flip sign if your convention differs.
        for ui, vi, zi in zip(u, v, z):
            if zi > depth[vi, ui]:
                depth[vi, ui] = zi

        # Replace empty with minimum
        empty = depth < -9000
        if np.all(empty):
            return np.zeros((H, W), dtype=np.uint8)

        dmin = np.min(depth[~empty])
        depth[empty] = dmin

        # Normalize to 0..255
        d_lo, d_hi = np.percentile(depth, 5), np.percentile(depth, 95)
        if d_hi - d_lo < 1e-6:
            return np.zeros((H, W), dtype=np.uint8)

        depth_norm = (depth - d_lo) / (d_hi - d_lo)
        depth_norm = np.clip(depth_norm, 0.0, 1.0)
        return (depth_norm * 255).astype(np.uint8)

    def embed_cluster(self, cluster: o3d.geometry.PointCloud) -> Optional[np.ndarray]:
        if not self.ok:
            return None

        pts = np.asarray(cluster.points, dtype=np.float32)
        if pts.shape[0] < 20:
            return None

        pts = self._sample_points(pts)
        pts = self._normalize_cluster(pts)

        # multi-view around z-axis
        angles = np.linspace(0, 2 * np.pi, num=self.n_views, endpoint=False)
        imgs = []
        for a in angles:
            p = self._rotate_z(pts, a)
            depth_u8 = self._render_depth_orthographic(p)

            # convert to 3-channel RGB image for CLIP preprocess
            # (depth repeated across channels)
            img3 = np.stack([depth_u8, depth_u8, depth_u8], axis=-1)
            imgs.append(img3)

        # Encode with CLIP
        import torch
        from PIL import Image

        emb_list = []
        with torch.no_grad():
            for im in imgs:
                pil = Image.fromarray(im)
                x = self._preprocess(pil).unsqueeze(0).to(self._device)
                feat = self._model.encode_image(x)  # [1, D]
                feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-12)
                emb_list.append(feat.squeeze(0).detach().cpu().numpy())

        if not emb_list:
            return None

        emb = np.mean(np.stack(emb_list, axis=0), axis=0)
        return l2_normalize(emb.astype(np.float32))


# =========================
# Detection
# =========================

class HumanDetector:
    """Detects human-like clusters from a point cloud."""

    def __init__(self):
        self.dbscan_eps = 0.5
        self.dbscan_min_points = 5

    def extract_shape_descriptor(self, cluster: o3d.geometry.PointCloud) -> np.ndarray:
        pts = np.asarray(cluster.points)
        if pts.shape[0] < 5:
            return np.zeros(4)

        centered = pts - pts.mean(axis=0)
        bbox = cluster.get_axis_aligned_bounding_box()
        ext = np.maximum(bbox.get_extent(), 1e-6)

        aspect_ratio = ext[2] / max(ext[0], ext[1])
        head_density = float(np.mean(pts[:, 2] > np.percentile(pts[:, 2], 80)))
        var = np.var(centered, axis=0)

        return np.array([aspect_ratio, head_density, var[0], var[2]], dtype=np.float32)

    def is_valid_human_geometry(self, cluster: o3d.geometry.PointCloud) -> bool:
        pts = np.asarray(cluster.points)
        if pts.shape[0] < 5:
            return False

        bbox = cluster.get_axis_aligned_bounding_box()
        ext = np.maximum(bbox.get_extent(), 1e-6)

        height = ext[2]
        width = max(ext[0], ext[1])
        density = pts.shape[0] / np.prod(ext)

        return (
            0.4 <= height <= 2.2 and
            width <= 1.0 and
            density > 5.0
        )

    def detect(self, pcd: o3d.geometry.PointCloud) -> List[Dict]:
        """Returns list of detections: {centroid, shape, cluster_pcd}"""
        if pcd.is_empty():
            return []

        try:
            pcd, _ = pcd.remove_statistical_outlier(20, 1.5)
        except RuntimeError:
            return []

        if len(pcd.points) < 3:
            return []

        try:
            _, inliers = pcd.segment_plane(0.15, 3, 1000)
            pcd = pcd.select_by_index(inliers, invert=True)
        except RuntimeError:
            pass

        labels = np.array(pcd.cluster_dbscan(self.dbscan_eps, self.dbscan_min_points))
        detections = []

        for label in np.unique(labels[labels >= 0]):
            cluster = pcd.select_by_index(np.where(labels == label)[0])
            if self.is_valid_human_geometry(cluster):
                detections.append({
                    "centroid": cluster.get_center(),
                    "shape": self.extract_shape_descriptor(cluster),
                    "cluster": cluster
                })

        return detections


# =========================
# Motion Model
# =========================

def build_imm_filter(initial_pos: np.ndarray) -> IMMEstimator:
    def base_kf(q_scale):
        kf = KalmanFilter(dim_x=6, dim_z=3)
        kf.F = np.eye(6)
        kf.H = np.block([[np.eye(3), np.zeros((3, 3))]])
        kf.x[:3] = initial_pos.reshape(3, 1)
        kf.P *= 5.0
        kf.R *= 10.0
        kf.Q = np.eye(6) * q_scale
        return kf

    kf_cv = base_kf(0.01)
    kf_rw = base_kf(0.1)

    mu = np.array([0.5, 0.5])
    trans = np.array([[0.95, 0.05], [0.05, 0.95]])

    return IMMEstimator([kf_cv, kf_rw], mu, trans)


# =========================
# Track
# =========================

class Track:
    def __init__(self, track_id: int, centroid: np.ndarray, shape: np.ndarray, emb: Optional[np.ndarray]):
        self.id = track_id
        self.kf = build_imm_filter(centroid)

        self.shape_history = [shape]
        self.emb_history = [emb] if emb is not None else []

        self.hits = 1
        self.age = 1
        self.skipped_frames = 0
        self.last_seen_pos = centroid
        self.is_static = False

    def predict(self, dt: float):
        for f in self.kf.filters:
            for i in range(3):
                f.F[i, i + 3] = dt
        self.kf.predict()
        self.age += 1

    def update(self, centroid: np.ndarray, shape: np.ndarray, emb: Optional[np.ndarray]):
        self.kf.update(centroid)
        self.shape_history.append(shape)
        if emb is not None:
            self.emb_history.append(emb)
        self.last_seen_pos = centroid
        self.skipped_frames = 0
        self.hits += 1

    def avg_shape(self) -> np.ndarray:
        return np.mean(self.shape_history[-15:], axis=0)

    def avg_emb(self) -> Optional[np.ndarray]:
        if not self.emb_history:
            return None
        return l2_normalize(np.mean(np.stack(self.emb_history[-10:], axis=0), axis=0))

    def velocity(self) -> np.ndarray:
        return self.kf.x[3:].flatten()

    def confidence(self, eps=1e-6, p_max=50.0) -> float:
        P = self.kf.P[:3, :3]
        unc = np.sqrt(np.linalg.det(P) + eps)
        return float(np.clip(np.exp(-unc / p_max), 0, 1))


# =========================
# Tracker
# =========================

class HumanTrackerMOT:
    def __init__(self):
        self.tracks: List[Track] = []
        self.lost_tracks: List[Track] = []
        self.next_id = 1
        self.history = []

        self.detector = HumanDetector()

        # ReID embedder (CLIP)
        self.embedder = ClipReIDEmbedder(
            enabled=True,            # set False to disable embeddings
            model_name="ViT-B-32",    # off-the-shelf CLIP
            pretrained="openai",
            image_size=224,
            n_views=6,
            points_per_cluster=1024,
        )

        self.min_hits = 3
        self.max_skip_dynamic = 20
        self.max_skip_static = 50

        # cost weights
        self.dist_weight = 0.55
        self.shape_weight = 0.10
        self.emb_weight = 0.35  # learned embedding weight

        # gating
        self.gating_threshold = 1.2

        # re-id thresholds
        self.reid_spatial_thresh = 1.5
        self.reid_emb_thresh = 0.35  # cosine distance threshold (lower is more similar)

    def _det_embedding(self, det: Dict) -> Optional[np.ndarray]:
        if not self.embedder.ok:
            return None
        try:
            return self.embedder.embed_cluster(det["cluster"])
        except Exception:
            return None

    def associate(self, detections: List[Dict]):
        if not self.tracks or not detections:
            return {}, set()

        # Precompute embeddings for detections (once)
        for d in detections:
            if "emb" not in d:
                d["emb"] = self._det_embedding(d)

        cost = np.full((len(self.tracks), len(detections)), 999.0, dtype=np.float32)

        for i, t in enumerate(self.tracks):
            tpos = t.kf.x[:3].flatten()
            temb = t.avg_emb()

            for j, d in enumerate(detections):
                dist = np.linalg.norm(tpos - d["centroid"])
                if dist > self.gating_threshold:
                    continue

                shape_dist = np.linalg.norm(t.avg_shape() - d["shape"])

                # embedding distance (cosine)
                emb_cost = 0.0
                if temb is not None and d.get("emb") is not None:
                    emb_cost = cosine_distance(temb, d["emb"])
                else:
                    # if missing embeddings, rely more on geometry
                    emb_cost = 0.75

                cost[i, j] = (
                    self.dist_weight * dist +
                    self.shape_weight * shape_dist +
                    self.emb_weight * emb_cost
                )

        rows, cols = linear_sum_assignment(cost)
        matches = {}

        for r, c in zip(rows, cols):
            # keep if within reasonable overall cost and spatially gated
            if cost[r, c] < self.gating_threshold:
                matches[r] = c

        return matches, set(matches.values())

    def _try_reid_from_lost(self, det: Dict) -> Optional[Track]:
        """
        Re-identification of a new detection against lost tracks using:
        - spatial proximity first
        - then embedding similarity if available
        """
        dpos = det["centroid"]
        demb = det.get("emb", None)

        best = None
        best_score = 999.0

        for lt in self.lost_tracks:
            spatial = np.linalg.norm(lt.last_seen_pos - dpos)
            if spatial > self.reid_spatial_thresh:
                continue

            lemb = lt.avg_emb()
            if lemb is not None and demb is not None:
                embd = cosine_distance(lemb, demb)
                score = 0.6 * spatial + 0.4 * embd
                if embd < self.reid_emb_thresh and score < best_score:
                    best = lt
                    best_score = score
            else:
                # fallback if embeddings missing
                score = spatial
                if score < best_score:
                    best = lt
                    best_score = score

        return best

    def update(self, frame_id: int, timestamp_ms: int, pcd_path: str):
        if not os.path.exists(pcd_path):
            return

        pcd = o3d.io.read_point_cloud(pcd_path)
        detections = self.detector.detect(pcd)

        # Compute embeddings for detections (once per frame)
        if self.embedder.ok:
            for d in detections:
                d["emb"] = self._det_embedding(d)
        else:
            for d in detections:
                d["emb"] = None

        prev_ts = self.history[-1]["timestamp_ms"] if self.history else timestamp_ms - 33
        dt = max((timestamp_ms - prev_ts) / 1000.0, 1e-3)

        for t in self.tracks:
            t.predict(dt)

        matches, used_dets = self.associate(detections)

        for ti, di in matches.items():
            d = detections[di]
            self.tracks[ti].update(d["centroid"], d["shape"], d.get("emb"))

        active = []
        for i, t in enumerate(self.tracks):
            if i not in matches:
                t.skipped_frames += 1

            speed = np.linalg.norm(t.velocity())
            t.is_static = speed < 0.15
            limit = self.max_skip_static if t.is_static else self.max_skip_dynamic

            if t.skipped_frames <= limit:
                active.append(t)
            else:
                self.lost_tracks.append(t)

        self.tracks = active

        # Create / ReID new tracks for unused detections
        for j, d in enumerate(detections):
            if j in used_dets:
                continue

            lt = self._try_reid_from_lost(d)
            if lt is not None:
                lt.kf.x[:3] = d["centroid"].reshape(3, 1)
                lt.skipped_frames = 0
                lt.update(d["centroid"], d["shape"], d.get("emb"))
                self.tracks.append(lt)
                self.lost_tracks.remove(lt)
            else:
                self.tracks.append(Track(self.next_id, d["centroid"], d["shape"], d.get("emb")))
                self.next_id += 1

        # Emit results (UNCHANGED schema)
        results = []
        for t in self.tracks:
            if t.skipped_frames == 0 and t.hits >= self.min_hits:
                pos = t.kf.x[:3].flatten()
                vel = t.velocity()
                results.append({
                    "id": t.id,
                    "position": [round(float(p), 3) for p in pos],
                    "velocity": [round(float(v), 3) for v in vel],
                    "speed": round(float(np.linalg.norm(vel)), 3),
                    "status": "STATIC" if t.is_static else "MOVING",
                    "confidence": round(t.confidence(), 3)
                })

        self.history.append({
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "detections": results
        })

    def finalize_results(self, output="tracking_results.json"):
        id_counts = {}
        for f in self.history:
            for d in f["detections"]:
                id_counts[d["id"]] = id_counts.get(d["id"], 0) + 1

        valid = {k for k, v in id_counts.items() if v > 10}
        final = []

        for f in self.history:
            dets = [d for d in f["detections"] if d["id"] in valid]
            if dets:
                f["detections"] = dets
                final.append(f)

        with open(output, "w") as f:
            json.dump(final, f, indent=4)

        print(f"✅ Tracking Finished. Results in {output}")


# =========================
# CLI
# =========================

def get_user_choice():
    print("\nChoose Dataset:\n [1] Human Only\n [2] Entire Map")
    while True:
        c = input("Enter 1 or 2: ").strip()
        if c == "1":
            return "mapHumanOnly"
        if c == "2":
            return "mapAll"


def main():
    tracker = HumanTrackerMOT()
    base = os.path.dirname(os.path.abspath(__file__))
    dataset = get_user_choice()
    data_dir = os.path.join(base, dataset)

    files = sorted(
        [f for f in os.listdir(data_dir) if f.endswith(".pcd")],
        key=extract_timestamp_ms
    )

    for i, f in enumerate(files):
        ts = extract_timestamp_ms(f)
        tracker.update(i, ts, os.path.join(data_dir, f))
        if i % 50 == 0:
            print(f"Processing frame {i}/{len(files)}")

    tracker.finalize_results()


if __name__ == "__main__":
    main()
