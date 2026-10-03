import os, time, queue, threading
from datetime import datetime
import cv2
import numpy as np
import freenect

RUN_DIR = os.path.join("tum_dataset", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(os.path.join(RUN_DIR, "rgb"), exist_ok=True)
os.makedirs(os.path.join(RUN_DIR, "depth"), exist_ok=True)

DEPTH_FACTOR = 5                         # mm -> TUM units (5000 per meter)
PNG = [cv2.IMWRITE_PNG_COMPRESSION, 1]   # fast compression
PREVIEW_INTERVAL = 0.1                   # seconds between preview refreshes

q = queue.Queue(maxsize=1200)
counts = {"rgb": 0, "depth": 0}
dropped = {"rgb": 0, "depth": 0}
latest_rgb = None
last_preview = 0.0

def worker():
    while True:
        item = q.get()
        if item is None:
            break
        kind, ts, data = item
        if kind == "rgb":
            cv2.imwrite(f"{RUN_DIR}/rgb/{ts}.png",
                        cv2.cvtColor(data, cv2.COLOR_RGB2BGR), PNG)
        else:
            cv2.imwrite(f"{RUN_DIR}/depth/{ts}.png",
                        (data.astype(np.uint32) * DEPTH_FACTOR).astype(np.uint16), PNG)

workers = [threading.Thread(target=worker) for _ in range(3)]
for w in workers:
    w.start()

# def enqueue(kind, data):
#     ts = f"{time.time():.6f}"
#     try:
#         q.put_nowait((kind, ts, data.copy()))
#         counts[kind] += 1
#         return True
#     except queue.Full:
#         dropped[kind] += 1
#         return False

def enqueue(kind, data):
    ts = f"{time.time():.6f}"
    frame = data.copy()
    try:
        q.put_nowait((kind, ts, frame))
        counts[kind] += 1
        return frame
    except queue.Full:
        dropped[kind] += 1
        return None

def depth_cb(dev, data, _):
    enqueue("depth", data)

# def rgb_cb(dev, data, _):
#     global latest_rgb
#     if enqueue("rgb", data):
#         latest_rgb = data            # enqueue already copied; this is only read by body()

def rgb_cb(dev, data, _):
    global latest_rgb
    frame = enqueue("rgb", data)
    if frame is not None:
        latest_rgb = frame

def body(*args):
    """Runs in the main loop: safe place to show the preview and stop."""
    global last_preview
    now = time.time()
    if latest_rgb is not None and now - last_preview > PREVIEW_INTERVAL:
        last_preview = now
        cv2.imshow("RGB (click window, press q to stop)",
                   cv2.cvtColor(latest_rgb, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise freenect.Kill

ctx = freenect.init()
dev = freenect.open_device(ctx, 0)
freenect.set_depth_mode(dev, freenect.RESOLUTION_MEDIUM, freenect.DEPTH_REGISTERED)
freenect.set_video_mode(dev, freenect.RESOLUTION_MEDIUM, freenect.VIDEO_RGB)

print(f"[INFO] Recording to {RUN_DIR}/  (q in preview window, or Ctrl+C, to stop)")
t0 = time.time()
try:
    freenect.runloop(dev=dev, depth=depth_cb, video=rgb_cb, body=body)
except (KeyboardInterrupt, freenect.Kill):
    pass
finally:
    dur = time.time() - t0
    cv2.destroyAllWindows()
    print("[INFO] Flushing frames to disk...")
    for _ in workers:
        q.put(None)
    for w in workers:
        w.join()

    freenect.close_device(dev)
    freenect.shutdown(ctx)
    
    print(f"[DONE] {dur:.1f}s | rgb {counts['rgb']/dur:.1f} fps | "
          f"depth {counts['depth']/dur:.1f} fps | dropped {dropped}")
    print(f"Next: python associate.py {RUN_DIR}")