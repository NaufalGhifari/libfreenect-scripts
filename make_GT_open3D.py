import sys
import numpy as np
import cv2
import open3d as o3d
from scipy.spatial.transform import Rotation as R

root = sys.argv[1]
W, H = 640, 480
fx, fy, cx, cy = 530.669, 528.217, 300.313, 270.653
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
dist = np.array([0.28678, -1.15652, 0.00102, -0.00207, 2.31469])

DEPTH_SCALE = 5000.0     # matches DEPTH_FACTOR = 5 in record.py (mm * 5)
DEPTH_TRUNC = 4.0        # metres
STRIDE = 1               # raise to 2-3 if odometry fails on near-identical frames

# Undistort maps (same K kept, so the pinhole intrinsics stay valid)
map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, K, (W, H), cv2.CV_32FC1)

intr = o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy)

# associations.txt: ts_rgb rgb/xxx.png ts_depth depth/xxx.png
assoc = [l.split() for l in open(f"{root}/associations.txt")
         if l.strip() and not l.startswith("#")]
assoc = assoc[::STRIDE]

def load(i):
    rgb = cv2.imread(f"{root}/{assoc[i][1]}", cv2.IMREAD_COLOR)
    depth = cv2.imread(f"{root}/{assoc[i][3]}", cv2.IMREAD_UNCHANGED)   # uint16
    rgb = cv2.remap(rgb, map1, map2, cv2.INTER_LINEAR)
    depth = cv2.remap(depth, map1, map2, cv2.INTER_NEAREST)             # no blending with 0s
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(np.ascontiguousarray(rgb)),
        o3d.geometry.Image(np.ascontiguousarray(depth)),
        depth_scale=DEPTH_SCALE, depth_trunc=DEPTH_TRUNC,
        convert_rgb_to_intensity=True)

opt = o3d.pipelines.odometry.OdometryOption()
jac = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

pose = np.eye(4)                      # camera-to-world; frame 0 defines the origin
poses = [(assoc[0][0], pose.copy())]
prev = load(0)
failures = 0

for i in range(1, len(assoc)):
    cur = load(i)
    ok, T, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
        cur, prev, intr, np.eye(4), jac, opt)     # source = cur, target = prev
    if not ok:
        failures += 1
        print(f"[WARN] odometry failed at frame {i} (ts {assoc[i][0]}), skipped")
        continue                                   # keep 'prev' so the next frame retries against it
    pose = pose @ T
    poses.append((assoc[i][0], pose.copy()))
    prev = cur
    if i % 100 == 0:
        print(f"{i}/{len(assoc)}")

with open(f"{root}/groundtruth.txt", "w") as f:
    f.write("# GT generated using Open3D\n# timestamp tx ty tz qx qy qz qw\n")
    for ts, P in poses:
        qx, qy, qz, qw = R.from_matrix(P[:3, :3]).as_quat()   # scipy order is x,y,z,w
        tx, ty, tz = P[:3, 3]
        f.write(f"{ts} {tx:.6f} {ty:.6f} {tz:.6f} {qx:.6f} {qy:.6f} {qz:.6f} {qw:.6f}\n")

print(f"Wrote {len(poses)} poses, {failures} failures -> {root}/groundtruth.txt")