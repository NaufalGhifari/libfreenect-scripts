import os, sys, shutil, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("run_dir")
ap.add_argument("--max_diff", type=float, default=0.02, help="max time gap in seconds (default 0.02)")
ap.add_argument("--delete", action="store_true", help="delete unmatched frames instead of moving them")
args = ap.parse_args()

run = args.run_dir.rstrip("/")
name = os.path.basename(run)

def load(kind):
    d = os.path.join(run, kind)
    files = [f for f in os.listdir(d) if f.endswith(".png")]
    items = sorted((float(os.path.splitext(f)[0]), f"{kind}/{f}") for f in files)
    return items

rgb, depth = load("rgb"), load("depth")
if not rgb or not depth:
    sys.exit("No rgb or depth frames found.")

t_rgb = np.array([t for t, _ in rgb])
t_dep = np.array([t for t, _ in depth])

# candidate pairs: the two nearest depth frames for each RGB frame
idx = np.searchsorted(t_dep, t_rgb)
cands = []
for i, k in enumerate(idx):
    for j in (k - 1, k):
        if 0 <= j < len(t_dep):
            d = abs(t_rgb[i] - t_dep[j])
            if d < args.max_diff:
                cands.append((d, i, j))
cands.sort()

used_i, used_j, matches = set(), set(), []
for d, i, j in cands:
    if i in used_i or j in used_j:
        continue
    used_i.add(i); used_j.add(j)
    matches.append((i, j, d))
matches.sort()

# write index files (matched frames only)
with open(os.path.join(run, "rgb.txt"), "w") as fr, \
     open(os.path.join(run, "depth.txt"), "w") as fd, \
     open(os.path.join(run, "associations.txt"), "w") as fa, \
     open(os.path.join(run, "groundtruth.txt"), "w") as fg:
    fr.write(f"# color images\n# file: '{name}'\n# timestamp filename\n")
    fd.write(f"# depth maps\n# file: '{name}'\n# timestamp filename\n")
    fg.write("# ground truth trajectory (DUMMY)\n# timestamp tx ty tz qx qy qz qw\n")
    for i, j, _ in matches:
        (tr, pr), (td, pd) = rgb[i], depth[j]
        fr.write(f"{tr:.6f} {pr}\n")
        fd.write(f"{td:.6f} {pd}\n")
        fa.write(f"{tr:.6f} {pr} {td:.6f} {pd}\n")
        fg.write(f"{tr:.6f} 0.0 0.0 0.0 0.0 0.0 0.0 1.0\n")

# handle unmatched frames
unmatched = [rgb[i][1] for i in range(len(rgb)) if i not in used_i] + \
            [depth[j][1] for j in range(len(depth)) if j not in used_j]
for rel in unmatched:
    src = os.path.join(run, rel)
    if args.delete:
        os.remove(src)
    else:
        dst = os.path.join(run, "unmatched", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.move(src, dst)

gaps = np.array([m[2] for m in matches]) * 1000
dur = t_rgb[-1] - t_rgb[0]
print(f"RGB frames: {len(rgb)} | depth frames: {len(depth)} | matched pairs: {len(matches)}")
print(f"Unmatched: {len(unmatched)} ({'deleted' if args.delete else 'moved to unmatched/'})")
if len(gaps):
    print(f"Time gap: mean {gaps.mean():.1f} ms, max {gaps.max():.1f} ms")
    print(f"Effective rate: {len(matches)/dur:.1f} pairs/s over {dur:.1f} s")