import cv2, numpy as np, glob

files = sorted(glob.glob("calib/*.png"))
print("images found:", len(files))

pattern = (8, 5)   # inner corners (columns, rows)
for f in files[:10]:
    g = cv2.imread(f, 0)
    if g is None:
        print(f, "-> could not read"); continue
    ok, _ = cv2.findChessboardCorners(g, pattern)
    print(f, g.shape, "found" if ok else "NOT found")


#pattern = (9, 6)            # inner corners of your board
sq = 0.025                  # square size in metres
objp = np.zeros((pattern[0]*pattern[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2) * sq

obj, img = [], []
for f in glob.glob("calib/*.png"):
    g = cv2.imread(f, 0)
    ok, c = cv2.findChessboardCorners(g, pattern)
    if ok:
        c = cv2.cornerSubPix(g, c, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
        obj.append(objp); img.append(c)

rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, g.shape[::-1], None, None)
print("RMS reprojection error:", rms)
print("K =\n", K, "\ndist =", dist.ravel())
