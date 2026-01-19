"""
tracking.py
-----------
Robust multi-object human tracking in 3D point clouds.

Off-the-shelf "SOTA-ish" improvements (no training, sensor-agnostic):
- Detection: voxel downsample + (optional) plane removal + DBSCAN + geometry checks
- Tracking: IMM with proper CV + RW models, dt-aware, physically consistent Q
- Association: two-stage (Mahalanobis Hungarian + appearance refinement)
- Re-ID: CLIP multi-view projection embeddings + Open3D FPFH descriptor
- Track-level embedding gallery (EMA + short-term buffer)
- Output: tracking_results.json (same schema)

Notes:
- True SOTA 3D ReID typically requires training. This is a very strong no-training baseline.
"""

import os
import re
import json
import numpy as np
import open3d as o3d
from typing import List, Dict, Optional, Tuple
from scipy.optimize import linear_sum_assignment
from filterpy.kalman import KalmanFilter, IMMEstimator

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


# =========================
# Utility
# =========================

def extract_timestamp_ms(filename: str) -> int:
    m = re.search(r"(\d+)ms", filename)
    if not m:
        raise ValueError(f"Invalid filename timestamp: {filename}")
    return int(m.group(1))


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(x) + eps)
    return (x / n).astype(np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    da = float(np.linalg.norm(a) + eps)
    db = float(np.linalg.norm(b) + eps)
    return float(1.0 - float(np.dot(a, b)) / (da * db))


def safe_det_3x3(M: np.ndarray, eps: float = 1e-12) -> float:
    try:
        return float(np.linalg.det(M) + eps)
    except Exception:
        return eps


# =========================
# Learned ReID Embeddings (CLIP projections)
# =========================

class ClipReIDEmbedder:
    """
    Off-the-shelf learned embedding for a 3D cluster using CLIP image encoder:
    - normalize cluster
    - render multi-view orthographic maps (depth + density)
    - encode with CLIP
    - average and L2-normalize

    Works without training, sensor-agnostic.
    """
    def __init__(
        self,
        enabled: bool = True,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        image_size: int = 224,
        n_views: int = 8,
        points_per_cluster: int = 2048,
        xy_clip: float = 1.3,
        z_clip: Tuple[float, float] = (-2.5, 2.5),
    ):
        self.enabled = enabled
        self.image_size = int(image_size)
        self.n_views = int(n_views)
        self.points_per_cluster = int(points_per_cluster)
        self.xy_clip = float(xy_clip)
        self.z_clip = (float(z_clip[0]), float(z_clip[1]))

        self._ok = False
        self._device = "cpu"
        self._model = None
        self._preprocess = None

        if not enabled:
            return

        try:
            import torch
            import open_clip
            from PIL import Image  # noqa

            self._torch = torch
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained
            )
            model.eval().to(self._device)

            self._model = model
            self._preprocess = preprocess
            self._ok = True

            print(f"✅ ReID enabled: CLIP {model_name} ({pretrained}) on {self._device}")

        except Exception as e:
            self._ok = False
            print(f"⚠️ ReID disabled (open_clip init failed): {e}")

    @property
    def ok(self) -> bool:
        return bool(self._ok)

    def _sample_points(self, pts: np.ndarray) -> np.ndarray:
        n = pts.shape[0]
        if n == 0:
            return pts
        if n >= self.points_per_cluster:
            idx = np.random.choice(n, self.points_per_cluster, replace=False)
            return pts[idx]
        idx = np.random.choice(n, self.points_per_cluster, replace=True)
        return pts[idx]

    def _normalize_cluster(self, pts: np.ndarray) -> np.ndarray:
        # center
        pts = pts - pts.mean(axis=0, keepdims=True)
        # robust scale to unit-ish radius
        r = np.linalg.norm(pts, axis=1)
        s = np.percentile(r, 90) + 1e-6
        return pts / s

    def _rotate_z(self, pts: np.ndarray, ang: float) -> np.ndarray:
        ca, sa = np.cos(ang), np.sin(ang)
        R = np.array([[ca, -sa, 0.0],
                      [sa,  ca, 0.0],
                      [0.0, 0.0, 1.0]], dtype=np.float32)
        return pts @ R.T

    def _render_maps(self, pts: np.ndarray) -> np.ndarray:
        """
        Render a 3-channel image:
        - channel 0: depth map (max z per pixel)
        - channel 1: density map (counts per pixel)
        - channel 2: "height mask" (fraction of points above z80 per pixel)
        """
        H = W = self.image_size
        x = np.clip(pts[:, 0], -self.xy_clip, self.xy_clip)
        y = np.clip(pts[:, 1], -self.xy_clip, self.xy_clip)
        z = np.clip(pts[:, 2], self.z_clip[0], self.z_clip[1])

        u = ((x + self.xy_clip) / (2 * self.xy_clip) * (W - 1)).astype(np.int32)
        v = ((self.xy_clip - y) / (2 * self.xy_clip) * (H - 1)).astype(np.int32)

        depth = np.full((H, W), -9999.0, dtype=np.float32)
        dens = np.zeros((H, W), dtype=np.float32)
        high = np.zeros((H, W), dtype=np.float32)

        z80 = np.percentile(z, 80)

        for ui, vi, zi in zip(u, v, z):
            dens[vi, ui] += 1.0
            if zi > depth[vi, ui]:
                depth[vi, ui] = zi
            if zi > z80:
                high[vi, ui] += 1.0

        # fill empties
        empty = depth < -9000
        if np.all(empty):
            return np.zeros((H, W, 3), dtype=np.uint8)

        dmin = np.min(depth[~empty])
        depth[empty] = dmin

        # normalize depth to [0,255]
        d_lo, d_hi = np.percentile(depth, 5), np.percentile(depth, 95)
        if d_hi - d_lo < 1e-6:
            depth_u8 = np.zeros((H, W), dtype=np.uint8)
        else:
            depth_n = np.clip((depth - d_lo) / (d_hi - d_lo), 0.0, 1.0)
            depth_u8 = (depth_n * 255).astype(np.uint8)

        # density normalization (log)
        dens_n = np.log1p(dens)
        if dens_n.max() > 0:
            dens_n = dens_n / dens_n.max()
        dens_u8 = (np.clip(dens_n, 0.0, 1.0) * 255).astype(np.uint8)

        # high mask: fraction high among density
        frac = np.zeros_like(high)
        nz = dens > 0
        frac[nz] = high[nz] / dens[nz]
        frac_u8 = (np.clip(frac, 0.0, 1.0) * 255).astype(np.uint8)

        return np.stack([depth_u8, dens_u8, frac_u8], axis=-1)

    def embed_cluster(self, cluster: o3d.geometry.PointCloud) -> Optional[np.ndarray]:
        if not self.ok:
            return None

        pts = np.asarray(cluster.points, dtype=np.float32)
        if pts.shape[0] < 30:
            return None

        pts = self._sample_points(pts)
        pts = self._normalize_cluster(pts)

        angles = np.linspace(0, 2 * np.pi, num=self.n_views, endpoint=False)

        import torch
        from PIL import Image

        embs = []
        with torch.no_grad():
            for a in angles:
                p = self._rotate_z(pts, float(a))
                img = self._render_maps(p)
                pil = Image.fromarray(img)
                x = self._preprocess(pil).unsqueeze(0).to(self._device)
                feat = self._model.encode_image(x)  # [1, D]
                feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-12)
                embs.append(feat.squeeze(0).detach().cpu().numpy())

        if not embs:
            return None

        emb = np.mean(np.stack(embs, axis=0), axis=0).astype(np.float32)
        return l2_normalize(emb)


# =========================
# 3D Geometry Descriptor (FPFH)
# =========================

class FPFHDescriptor:
    """
    Robust handcrafted descriptor used as a complementary cue.
    Helps when CLIP projections struggle (e.g., sparse clusters).
    """
    def __init__(self, enabled: bool = True, voxel: float = 0.10):
        self.enabled = enabled
        self.voxel = float(voxel)

    def compute(self, cluster: o3d.geometry.PointCloud) -> Optional[np.ndarray]:
        if not self.enabled:
            return None
        if cluster.is_empty() or len(cluster.points) < 50:
            return None
        try:
            p = cluster.voxel_down_sample(self.voxel)
            if len(p.points) < 30:
                return None

            p.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel * 2.5, max_nn=60
                )
            )
            fpfh = o3d.pipelines.registration.compute_fpfh_feature(
                p,
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.voxel * 5.0, max_nn=120
                )
            )
            # fpfh.data shape: (33, N)
            v = np.mean(np.asarray(fpfh.data), axis=1).astype(np.float32)  # (33,)
            return l2_normalize(v)
        except Exception:
            return None


# =========================
# Detection
# =========================

class HumanDetector:
    """Detects human-like clusters from a point cloud."""

    def __init__(self):
        # Detection tuning (sensor-agnostic defaults)
        self.voxel_size = 0.07
        self.remove_ground = True

        self.dbscan_eps = 0.45
        self.dbscan_min_points = 8

        # geometry constraints (relaxed but still human-ish)
        self.min_height = 0.5
        self.max_height = 2.3
        self.max_width = 1.2

        self.min_density = 2.0  # relaxed: sensors vary a lot

    def extract_shape_descriptor(self, cluster: o3d.geometry.PointCloud) -> np.ndarray:
        pts = np.asarray(cluster.points)
        if pts.shape[0] < 10:
            return np.zeros(6, dtype=np.float32)

        bbox = cluster.get_axis_aligned_bounding_box()
        ext = np.maximum(bbox.get_extent(), 1e-6)
        centered = pts - pts.mean(axis=0)
        var = np.var(centered, axis=0)

        # richer shape vector than your original
        height = ext[2]
        width = max(ext[0], ext[1])
        aspect_hw = height / width
        density = float(pts.shape[0] / np.prod(ext))

        top_frac = float(np.mean(pts[:, 2] > np.percentile(pts[:, 2], 80)))

        return np.array(
            [height, width, aspect_hw, density, var[0], var[2] + top_frac],
            dtype=np.float32
        )

    def is_valid_human_geometry(self, cluster: o3d.geometry.PointCloud) -> bool:
        pts = np.asarray(cluster.points)
        if pts.shape[0] < 30:
            return False

        bbox = cluster.get_axis_aligned_bounding_box()
        ext = np.maximum(bbox.get_extent(), 1e-6)
        height = ext[2]
        width = max(ext[0], ext[1])
        density = float(pts.shape[0] / np.prod(ext))

        if not (self.min_height <= height <= self.max_height):
            return False
        if width > self.max_width:
            return False
        if density < self.min_density:
            return False

        return True

    def _preprocess(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
        if pcd.is_empty():
            return pcd

        # voxel downsample helps across unknown sensors
        try:
            p = pcd.voxel_down_sample(self.voxel_size)
        except Exception:
            p = pcd

        # remove outliers
        try:
            p, _ = p.remove_statistical_outlier(nb_neighbors=20, std_ratio=1.7)
        except Exception:
            pass

        # optional ground removal
        if self.remove_ground and len(p.points) > 50:
            try:
                # plane: distance threshold tied to voxel
                _, inliers = p.segment_plane(
                    distance_threshold=max(0.12, self.voxel_size * 2.0),
                    ransac_n=3,
                    num_iterations=1000
                )
                p = p.select_by_index(inliers, invert=True)
            except Exception:
                pass

        return p

    def detect(self, pcd: o3d.geometry.PointCloud) -> List[Dict]:
        """Returns detections: {centroid, shape, cluster}"""
        p = self._preprocess(pcd)
        if p.is_empty() or len(p.points) < 30:
            return []

        labels = np.array(p.cluster_dbscan(self.dbscan_eps, self.dbscan_min_points))
        detections: List[Dict] = []

        valid_labels = np.unique(labels[labels >= 0])
        for lb in valid_labels:
            idx = np.where(labels == lb)[0]
            cluster = p.select_by_index(idx)
            if self.is_valid_human_geometry(cluster):
                detections.append({
                    "centroid": cluster.get_center(),
                    "shape": self.extract_shape_descriptor(cluster),
                    "cluster": cluster
                })

        return detections


# =========================
# Motion Model (IMM: CV + RW)
# =========================

def build_cv_kf(initial_pos: np.ndarray, dt: float) -> KalmanFilter:
    """
    Constant velocity model:
    x = [px, py, pz, vx, vy, vz]
    """
    kf = KalmanFilter(dim_x=6, dim_z=3)

    F = np.eye(6, dtype=np.float32)
    F[0, 3] = dt
    F[1, 4] = dt
    F[2, 5] = dt
    kf.F = F

    kf.H = np.block([[np.eye(3, dtype=np.float32), np.zeros((3, 3), dtype=np.float32)]])

    kf.x = np.zeros((6, 1), dtype=np.float32)
    kf.x[:3, 0] = initial_pos.astype(np.float32)

    # uncertainty
    kf.P = np.eye(6, dtype=np.float32) * 5.0

    # measurement noise (sensor unknown -> fairly high)
    kf.R = np.eye(3, dtype=np.float32) * 0.25

    # process noise from continuous white noise acceleration
    # q is accel std (m/s^2) unknown -> use moderate
    q = 2.0
    dt2 = dt * dt
    dt3 = dt2 * dt
    dt4 = dt2 * dt2

    Qpos = (dt4 / 4.0) * q * q
    Qcross = (dt3 / 2.0) * q * q
    Qvel = (dt2) * q * q

    Q = np.zeros((6, 6), dtype=np.float32)
    for a in range(3):
        Q[a, a] = Qpos
        Q[a, a + 3] = Qcross
        Q[a + 3, a] = Qcross
        Q[a + 3, a + 3] = Qvel

    kf.Q = Q
    return kf


def build_rw_kf(initial_pos: np.ndarray, dt: float) -> KalmanFilter:
    """
    Random-walk-ish model (more jitter allowed).
    Still uses same state but higher process noise and weaker velocity coupling.
    """
    kf = build_cv_kf(initial_pos, dt)

    # In RW, velocity is less reliable -> larger process noise
    kf.Q *= 4.0
    # Also larger measurement noise (model mismatch)
    kf.R *= 1.5
    return kf


def build_imm_filter(initial_pos: np.ndarray, dt: float) -> IMMEstimator:
    kf_cv = build_cv_kf(initial_pos, dt)
    kf_rw = build_rw_kf(initial_pos, dt)

    mu = np.array([0.7, 0.3], dtype=np.float32)  # prefer CV by default
    trans = np.array([[0.97, 0.03],
                      [0.06, 0.94]], dtype=np.float32)

    return IMMEstimator([kf_cv, kf_rw], mu, trans)


# =========================
# Track
# =========================

class Track:
    def __init__(
        self,
        track_id: int,
        centroid: np.ndarray,
        shape: np.ndarray,
        emb: Optional[np.ndarray],
        fpfh: Optional[np.ndarray],
        init_dt: float
    ):
        self.id = track_id
        self.kf = build_imm_filter(centroid, init_dt)

        self.shape_history = [shape]
        self.emb_history: List[np.ndarray] = []
        self.fpfh_history: List[np.ndarray] = []

        if emb is not None:
            self.emb_history.append(emb)
            self.emb_ema = emb.copy()
        else:
            self.emb_ema = None

        if fpfh is not None:
            self.fpfh_history.append(fpfh)
            self.fpfh_ema = fpfh.copy()
        else:
            self.fpfh_ema = None

        self.hits = 1
        self.age = 1
        self.skipped_frames = 0
        self.last_seen_pos = centroid.astype(np.float32)
        self.is_static = False

    def predict(self, dt: float):
        # update each filter's F and Q with dt
        # (FilterPy IMM holds filters we can edit)
        for f in self.kf.filters:
            # F
            f.F = np.eye(6, dtype=np.float32)
            f.F[0, 3] = dt
            f.F[1, 4] = dt
            f.F[2, 5] = dt

            # Q (same formula as CV builder, scaled by existing factor)
            # keep proportionality: derive base Q then scale by prior diagonal ratio
            q = 2.0
            dt2 = dt * dt
            dt3 = dt2 * dt
            dt4 = dt2 * dt2

            Qpos = (dt4 / 4.0) * q * q
            Qcross = (dt3 / 2.0) * q * q
            Qvel = (dt2) * q * q

            Q = np.zeros((6, 6), dtype=np.float32)
            for a in range(3):
                Q[a, a] = Qpos
                Q[a, a + 3] = Qcross
                Q[a + 3, a] = Qcross
                Q[a + 3, a + 3] = Qvel

            # preserve relative scale between CV/RW
            scale = float(np.mean(np.diag(f.Q)) / (np.mean(np.diag(Q)) + 1e-12))
            f.Q = Q * scale

        self.kf.predict()
        self.age += 1

    def update(self, centroid: np.ndarray, shape: np.ndarray, emb: Optional[np.ndarray], fpfh: Optional[np.ndarray]):
        self.kf.update(centroid.astype(np.float32))
        self.shape_history.append(shape.astype(np.float32))

        # EMA for embeddings improves robustness
        alpha = 0.25

        if emb is not None:
            self.emb_history.append(emb)
            if self.emb_ema is None:
                self.emb_ema = emb.copy()
            else:
                self.emb_ema = l2_normalize((1 - alpha) * self.emb_ema + alpha * emb)

        if fpfh is not None:
            self.fpfh_history.append(fpfh)
            if self.fpfh_ema is None:
                self.fpfh_ema = fpfh.copy()
            else:
                self.fpfh_ema = l2_normalize((1 - alpha) * self.fpfh_ema + alpha * fpfh)

        self.last_seen_pos = centroid.astype(np.float32)
        self.skipped_frames = 0
        self.hits += 1

    def avg_shape(self) -> np.ndarray:
        return np.mean(np.stack(self.shape_history[-15:], axis=0), axis=0)

    def velocity(self) -> np.ndarray:
        return self.kf.x[3:].flatten().astype(np.float32)

    def position(self) -> np.ndarray:
        return self.kf.x[:3].flatten().astype(np.float32)

    def covariance_pos(self) -> np.ndarray:
        return self.kf.P[:3, :3].astype(np.float32)

    def confidence(self, eps=1e-6, p_max=10.0) -> float:
        P = self.covariance_pos()
        unc = np.sqrt(max(safe_det_3x3(P, eps), eps))
        return float(np.clip(np.exp(-unc / p_max), 0.0, 1.0))


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

        # ReID cues
        self.clip = ClipReIDEmbedder(
            enabled=True,
            model_name="ViT-B-32",
            pretrained="openai",
            image_size=224,
            n_views=8,
            points_per_cluster=2048
        )
        self.fpfh = FPFHDescriptor(enabled=True, voxel=0.10)

        # Track management
        self.min_hits = 3
        self.max_skip_dynamic = 25
        self.max_skip_static = 60

        # Gating and association (statistical gating)
        # For 3D, chi-square threshold for 0.997 quantile ~ 16.27, for 0.99 ~ 11.34.
        # We'll use a strong-but-not-too-strict gate.
        self.maha_gate = 12.0

        # Stage-2 weights (appearance refinement)
        self.shape_weight = 0.10
        self.clip_weight = 0.55
        self.fpfh_weight = 0.25
        self.euclid_weight = 0.10

        # ReID thresholds (lost track recovery)
        self.reid_spatial_max = 2.2
        self.reid_clip_max = 0.35     # cosine distance
        self.reid_fpfh_max = 0.55     # cosine distance (less strict)

    def _compute_det_features(self, det: Dict):
        # CLIP emb
        if self.clip.ok:
            try:
                det["emb"] = self.clip.embed_cluster(det["cluster"])
            except Exception:
                det["emb"] = None
        else:
            det["emb"] = None

        # FPFH
        try:
            det["fpfh"] = self.fpfh.compute(det["cluster"])
        except Exception:
            det["fpfh"] = None

    def _mahalanobis(self, track: Track, z: np.ndarray) -> float:
        """
        Mahalanobis distance of measurement z to predicted state.
        S = HPH' + R (approx: use track.kf.P and filter's R; IMM doesn't expose a single R cleanly,
        so we approximate using the mix covariance P and a fixed R).
        """
        x = track.position()
        P = track.covariance_pos()
        R = np.eye(3, dtype=np.float32) * 0.25

        v = (z.astype(np.float32) - x).reshape(3, 1)
        S = P + R
        try:
            Sinv = np.linalg.inv(S)
        except Exception:
            return 999.0

        d2 = float((v.T @ Sinv @ v).squeeze())
        return d2

    def _stage1_motion_assignment(self, detections: List[Dict]) -> Tuple[Dict[int, int], set]:
        """
        First stage: Hungarian on Mahalanobis distances only.
        Strongly stabilizes IDs.
        """
        if not self.tracks or not detections:
            return {}, set()

        cost = np.full((len(self.tracks), len(detections)), 9999.0, dtype=np.float32)

        for i, t in enumerate(self.tracks):
            for j, d in enumerate(detections):
                d2 = self._mahalanobis(t, d["centroid"])
                if d2 <= self.maha_gate:
                    cost[i, j] = d2

        rows, cols = linear_sum_assignment(cost)
        matches = {}
        used = set()

        for r, c in zip(rows, cols):
            if cost[r, c] <= self.maha_gate:
                matches[r] = c
                used.add(c)

        return matches, used

    def _appearance_cost(self, t: Track, d: Dict) -> float:
        """
        Second-stage refinement cost:
        combine CLIP + FPFH + shape + euclid (all normalized-ish).
        """
        # euclid (bounded)
        eu = float(np.linalg.norm(t.position() - d["centroid"]))
        eu = min(eu / 2.5, 1.0)

        # shape distance (normalize by typical scale)
        sh = float(np.linalg.norm(t.avg_shape() - d["shape"]))
        sh = min(sh / 5.0, 1.0)

        # CLIP cosine
        if t.emb_ema is not None and d.get("emb") is not None:
            ce = cosine_distance(t.emb_ema, d["emb"])
        else:
            ce = 0.75  # missing -> penalize

        # FPFH cosine
        if t.fpfh_ema is not None and d.get("fpfh") is not None:
            fe = cosine_distance(t.fpfh_ema, d["fpfh"])
        else:
            fe = 0.75

        # adaptive weighting: if track confidence low, rely more on appearance
        conf = t.confidence()
        # when conf low -> increase appearance contribution
        clip_w = self.clip_weight * (1.0 + (1.0 - conf) * 0.35)
        fpfh_w = self.fpfh_weight * (1.0 + (1.0 - conf) * 0.25)

        total_w = (self.euclid_weight + self.shape_weight + clip_w + fpfh_w)
        return (
            self.euclid_weight * eu +
            self.shape_weight * sh +
            clip_w * ce +
            fpfh_w * fe
        ) / max(total_w, 1e-6)

    def _refine_with_appearance(self, detections: List[Dict], stage1_matches: Dict[int, int]) -> Dict[int, int]:
        """
        Optional refinement:
        For the already motion-matched pairs, we keep them.
        For unmatched tracks/dets, we run another Hungarian on appearance cost,
        but only if they pass motion gating.
        """
        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in stage1_matches]
        unmatched_dets = [j for j in range(len(detections)) if j not in stage1_matches.values()]

        if not unmatched_tracks or not unmatched_dets:
            return stage1_matches

        cost = np.full((len(unmatched_tracks), len(unmatched_dets)), 9999.0, dtype=np.float32)

        for a, ti in enumerate(unmatched_tracks):
            t = self.tracks[ti]
            for b, dj in enumerate(unmatched_dets):
                d = detections[dj]

                # still require motion gating
                d2 = self._mahalanobis(t, d["centroid"])
                if d2 > self.maha_gate:
                    continue

                cost[a, b] = self._appearance_cost(t, d)

        rows, cols = linear_sum_assignment(cost)

        matches = dict(stage1_matches)
        for r, c in zip(rows, cols):
            if cost[r, c] < 0.55:  # tuned: accept only fairly good appearance matches
                ti = unmatched_tracks[r]
                dj = unmatched_dets[c]
                matches[ti] = dj

        return matches

    def _try_reid_from_lost(self, det: Dict) -> Optional[Track]:
        """
        Re-identify a new detection against lost tracks using:
        - spatial proximity
        - CLIP and/or FPFH similarity when available
        """
        dpos = det["centroid"].astype(np.float32)
        demb = det.get("emb")
        dfpfh = det.get("fpfh")

        best = None
        best_score = 9999.0

        for lt in self.lost_tracks:
            spatial = float(np.linalg.norm(lt.last_seen_pos - dpos))
            if spatial > self.reid_spatial_max:
                continue

            clip_d = 0.75
            if lt.emb_ema is not None and demb is not None:
                clip_d = cosine_distance(lt.emb_ema, demb)

            fpfh_d = 0.75
            if lt.fpfh_ema is not None and dfpfh is not None:
                fpfh_d = cosine_distance(lt.fpfh_ema, dfpfh)

            # strict acceptance if we have embeddings
            ok_clip = (clip_d < self.reid_clip_max) if (lt.emb_ema is not None and demb is not None) else True
            ok_fpfh = (fpfh_d < self.reid_fpfh_max) if (lt.fpfh_ema is not None and dfpfh is not None) else True

            if not (ok_clip and ok_fpfh):
                continue

            # score combines spatial + descriptors
            score = 0.45 * (spatial / self.reid_spatial_max) + 0.40 * clip_d + 0.15 * fpfh_d
            if score < best_score:
                best = lt
                best_score = score

        return best

    def update(self, frame_id: int, timestamp_ms: int, pcd_path: str):
        if not os.path.exists(pcd_path):
            return

        pcd = o3d.io.read_point_cloud(pcd_path)
        detections = self.detector.detect(pcd)

        # compute dt
        prev_ts = self.history[-1]["timestamp_ms"] if self.history else (timestamp_ms - 33)
        dt = max((timestamp_ms - prev_ts) / 1000.0, 1e-3)

        # predict all tracks
        for t in self.tracks:
            t.predict(dt)

        # features for detections (once)
        for d in detections:
            self._compute_det_features(d)

        # stage 1: motion assignment
        stage1_matches, used = self._stage1_motion_assignment(detections)

        # stage 2: appearance refinement
        matches = self._refine_with_appearance(detections, stage1_matches)
        used = set(matches.values())

        # apply updates
        for ti, di in matches.items():
            d = detections[di]
            self.tracks[ti].update(
                d["centroid"].astype(np.float32),
                d["shape"].astype(np.float32),
                d.get("emb"),
                d.get("fpfh")
            )

        # manage skipped tracks
        active = []
        for i, t in enumerate(self.tracks):
            if i not in matches:
                t.skipped_frames += 1

            speed = float(np.linalg.norm(t.velocity()))
            t.is_static = speed < 0.12

            limit = self.max_skip_static if t.is_static else self.max_skip_dynamic

            if t.skipped_frames <= limit:
                active.append(t)
            else:
                self.lost_tracks.append(t)

        self.tracks = active

        # create/re-id new tracks for unused detections
        for j, d in enumerate(detections):
            if j in used:
                continue

            lt = self._try_reid_from_lost(d)
            if lt is not None:
                # revive lost track
                lt.kf.x[:3, 0] = d["centroid"].astype(np.float32)
                lt.skipped_frames = 0
                lt.update(d["centroid"], d["shape"], d.get("emb"), d.get("fpfh"))
                self.tracks.append(lt)
                self.lost_tracks.remove(lt)
            else:
                # new track
                self.tracks.append(
                    Track(
                        self.next_id,
                        d["centroid"].astype(np.float32),
                        d["shape"].astype(np.float32),
                        d.get("emb"),
                        d.get("fpfh"),
                        init_dt=dt
                    )
                )
                self.next_id += 1

        # output results (UNCHANGED schema)
        results = []
        for t in self.tracks:
            if t.skipped_frames == 0 and t.hits >= self.min_hits:
                pos = t.position()
                vel = t.velocity()
                results.append({
                    "id": t.id,
                    "position": [round(float(p), 3) for p in pos],
                    "velocity": [round(float(v), 3) for v in vel],
                    "speed": round(float(np.linalg.norm(vel)), 3),
                    "status": "STATIC" if t.is_static else "MOVING",
                    "confidence": round(float(t.confidence()), 3)
                })

        self.history.append({
            "frame_id": frame_id,
            "timestamp_ms": timestamp_ms,
            "detections": results
        })

    def finalize_results(self, output="tracking_results.json"):
        # keep only sufficiently long tracks (helps remove noise)
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

        with open(output, "w") as fp:
            json.dump(final, fp, indent=4)

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
