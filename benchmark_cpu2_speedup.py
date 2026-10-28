import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Bind to 2 CPU cores
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

if hasattr(os, "sched_setaffinity"):
    try:
        os.sched_setaffinity(0, {0, 1})
    except Exception as e:
        print("sched_setaffinity note:", e)

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

import mrcnn.model as modellib
from mrcnn.config import Config

TEST_IMG = "/home/adis_mayur/F_CODES/Mask_RCNN/test_samples/cow_video_frame_1.jpg"
WEIGHTS = "/home/adis_mayur/F_CODES/crop11dec.h5"

def run_test(name, min_dim, max_dim, rois_inference):
    print(f"\n--- Testing: {name} (Dim: {min_dim}x{max_dim}, ROIs: {rois_inference}) ---")
    
    class TestCfg(Config):
        NAME = "CowMuzzle"
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
        NUM_CLASSES = 1 + 1
        DETECTION_MIN_CONFIDENCE = 0.70
        DETECTION_NMS_THRESHOLD = 0.30
        IMAGE_MIN_DIM = min_dim
        IMAGE_MAX_DIM = max_dim
        IMAGE_RESIZE_MODE = "square"
        POST_NMS_ROIS_INFERENCE = rois_inference
        
    config = TestCfg()
    model = modellib.MaskRCNN(mode="inference", config=config, model_dir="/tmp")
    model.load_weights(WEIGHTS, by_name=True)
    
    # Warmup
    img = cv2.imread(TEST_IMG)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Measure inference
    t0 = time.perf_counter()
    molded_images, image_metas, windows = model.mold_inputs([rgb])
    molded_shape = molded_images[0].shape
    anchors = model.get_anchors(molded_shape)
    anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)
    
    t_fwd_0 = time.perf_counter()
    detections, _, _, mrcnn_mask, _, _, _ = model.keras_model.predict(
        [molded_images, image_metas, anchors], verbose=0
    )
    t_fwd = (time.perf_counter() - t_fwd_0) * 1000
    
    rois, class_ids, scores, masks = model.unmold_detections(
        detections[0], mrcnn_mask[0], rgb.shape, molded_shape, windows[0]
    )
    t_total = (time.perf_counter() - t0) * 1000
    
    det_count = len(scores[scores >= 0.7])
    top_score = scores[0] if len(scores) > 0 else 0
    top_box = rois[0] if len(rois) > 0 else []
    
    print(f"Results for {name}:")
    print(f"  Forward Pass Latency: {t_fwd:.1f} ms ({t_fwd/1000:.2f} s)")
    print(f"  Total Latency:        {t_total:.1f} ms ({t_total/1000:.2f} s)")
    print(f"  Muzzles Detected:     {det_count} (Top Conf: {top_score*100:.1f}%)")
    print(f"  Detected Box:         {top_box}")
    return t_fwd, top_score

if __name__ == "__main__":
    # Baseline
    t_base, s_base = run_test("Baseline", 800, 1024, 1000)
    # Reduced ROIs
    t_rois, s_rois = run_test("Reduced ROIs (100)", 800, 1024, 100)
    # 512x512 resolution + 100 ROIs
    t_512, s_512 = run_test("512x512 + 100 ROIs", 512, 512, 100)
    # 640x640 resolution + 100 ROIs
    t_640, s_640 = run_test("640x640 + 100 ROIs", 640, 640, 100)
    
    print("\n" + "="*70)
    print("SPEED COMPARISON ON 2 CPU CORES:")
    print(f"Baseline (1024x1024, 1000 ROIs): {t_base/1000:.2f}s | Conf: {s_base*100:.1f}%")
    print(f"Reduced ROIs (100 ROIs):         {t_rois/1000:.2f}s (Speedup: {t_base/t_rois:.2f}x)")
    print(f"640x640 + 100 ROIs:              {t_640/1000:.2f}s (Speedup: {t_base/t_640:.2f}x)")
    print(f"512x512 + 100 ROIs:              {t_512/1000:.2f}s (Speedup: {t_base/t_512:.2f}x)")
    print("="*70)
