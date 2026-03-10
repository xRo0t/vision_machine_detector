"""
Vision Machine Camera - Live Camera Defect Detection
=====================================================
Captures frames from a webcam and compares them against a reference image
in real-time using CNN + OpenCV. Runs fully automatically - no keyboard needed.

Close the window or press Ctrl+C to stop.

Usage:
    python camera.py --reference images/reference_straight.png
    python camera.py --reference images/reference_straight.png --camera 0
    python camera.py --reference images/reference_straight.png --interval 3
"""

import argparse
import sys
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from torchvision import models, transforms


# Configuration
CNN_SIMILARITY_THRESHOLD = 0.90
SSIM_THRESHOLD = 0.80
PERCEPTUAL_DIFF_THRESHOLD = 0.15
CONTOUR_DIFF_THRESHOLD = 5
MIN_CONTOUR_AREA = 1500
CNN_INPUT_SIZE = (224, 224)
WARNING_THRESHOLD = 2

# Reference localization inside the camera frame
REFERENCE_MATCH_THRESHOLD = 0.45
TEMPLATE_SCALE_STEPS = 7
MIN_TEMPLATE_SCALE_RATIO = 0.50
LIVE_BASELINE_FRAMES = 8
LIVE_REFERENCE_SIZE = (384, 384)
LIVE_REFERENCE_MIN_MATCH = 0.55
CORE_CROP_LEFT_RATIO = 0.12
CORE_CROP_RIGHT_RATIO = 0.12
CORE_CROP_TOP_RATIO = 0.18
CORE_CROP_BOTTOM_RATIO = 0.14


class CNNFeatureExtractor:
    def __init__(self):
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model.eval()
        self.feature_layers = torch.nn.Sequential(*list(self.model.children())[:-1])

        children = list(self.model.children())
        self.layer_names = ["layer1", "layer2", "layer3", "layer4"]
        self.layer_extractors = {}
        for i, name in enumerate(self.layer_names):
            self.layer_extractors[name] = torch.nn.Sequential(*children[: 4 + i + 1])

        self.preprocess = transforms.Compose(
            [
                transforms.Resize(CNN_INPUT_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def extract_features_from_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features = self.feature_layers(tensor)
        return features.flatten()

    def extract_layer_features_from_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = self.preprocess(img).unsqueeze(0)
        layer_features = {}
        with torch.no_grad():
            for name, extractor in self.layer_extractors.items():
                layer_features[name] = extractor(tensor)
        return layer_features

    def extract_features_from_path(self, image_path):
        img = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features = self.feature_layers(tensor)
        return features.flatten()

    def extract_layer_features_from_path(self, image_path):
        img = Image.open(image_path).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0)
        layer_features = {}
        with torch.no_grad():
            for name, extractor in self.layer_extractors.items():
                layer_features[name] = extractor(tensor)
        return layer_features

    def cosine_similarity(self, feat1, feat2):
        return F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()

    def perceptual_difference(self, layers1, layers2):
        total_diff = 0.0
        weights = {"layer1": 0.1, "layer2": 0.2, "layer3": 0.3, "layer4": 0.4}
        for name in self.layer_names:
            f1 = layers1[name].flatten()
            f2 = layers2[name].flatten()
            cos_sim = F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
            total_diff += weights[name] * (1.0 - cos_sim)
        return total_diff


def build_reference_templates(ref_gray, frame_size):
    """Create a small pyramid of reference templates that fit inside the camera frame."""
    frame_h, frame_w = frame_size
    ref_h, ref_w = ref_gray.shape[:2]
    ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 1.2)

    max_scale = min(frame_w / ref_w, frame_h / ref_h)
    min_scale = max(0.35, max_scale * MIN_TEMPLATE_SCALE_RATIO)

    if max_scale <= 0:
        return []

    if min_scale >= max_scale:
        scales = [max_scale]
    else:
        scales = np.linspace(min_scale, max_scale, TEMPLATE_SCALE_STEPS)

    templates = []
    seen_sizes = set()
    for scale in scales:
        tmpl_w = max(32, int(ref_w * scale))
        tmpl_h = max(32, int(ref_h * scale))
        if tmpl_w > frame_w or tmpl_h > frame_h:
            continue
        if (tmpl_w, tmpl_h) in seen_sizes:
            continue
        seen_sizes.add((tmpl_w, tmpl_h))
        templates.append(
            {
                "scale": float(scale),
                "scale_ratio": float(scale / max_scale),
                "size": (tmpl_w, tmpl_h),
                "image": cv2.resize(
                    ref_blur,
                    (tmpl_w, tmpl_h),
                    interpolation=cv2.INTER_AREA,
                ),
            }
        )

    return templates


def extract_core_pattern(image):
    """Focus comparison on the central pattern area and ignore screen/frame margins."""
    h, w = image.shape[:2]
    x1 = int(w * CORE_CROP_LEFT_RATIO)
    x2 = int(w * (1.0 - CORE_CROP_RIGHT_RATIO))
    y1 = int(h * CORE_CROP_TOP_RATIO)
    y2 = int(h * (1.0 - CORE_CROP_BOTTOM_RATIO))

    x1 = max(0, min(x1, w - 2))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 2))
    y2 = max(y1 + 1, min(y2, h))
    return image[y1:y2, x1:x2]


def build_reference_state(extractor, reference_bgr, frame_size):
    """Compute all reference artifacts once so the loop can reuse them."""
    analysis_image = extract_core_pattern(reference_bgr)
    ref_gray = cv2.cvtColor(analysis_image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
    return {
        "image": analysis_image,
        "gray": ref_gray,
        "features": extractor.extract_features_from_frame(analysis_image),
        "layers": extractor.extract_layer_features_from_frame(analysis_image),
        "templates": build_reference_templates(template_gray, frame_size),
        "thresholds": default_thresholds(),
    }


def build_live_reference(calibration_rois):
    """Fuse several camera ROIs into a single stable live reference."""
    if not calibration_rois:
        raise ValueError("calibration_rois must not be empty")

    stack = np.stack(
        [
            cv2.resize(roi, LIVE_REFERENCE_SIZE, interpolation=cv2.INTER_AREA)
            for roi in calibration_rois
        ]
    ).astype(np.float32)
    return np.median(stack, axis=0).astype(np.uint8)


def default_thresholds():
    return {
        "cnn": CNN_SIMILARITY_THRESHOLD,
        "ssim": SSIM_THRESHOLD,
        "perceptual": PERCEPTUAL_DIFF_THRESHOLD,
        "contour": CONTOUR_DIFF_THRESHOLD,
    }


def locate_reference_region(frame, templates, ref_size):
    """
    Find the reference-like window inside the live camera image.
    The detector then compares that localized ROI instead of the full frame.
    """
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_blur = cv2.GaussianBlur(frame_gray, (5, 5), 1.2)
    frame_h, frame_w = frame_gray.shape[:2]

    best_match = None
    for template in templates:
        tmpl = template["image"]
        tmpl_w, tmpl_h = template["size"]

        corr_map = cv2.matchTemplate(frame_blur, tmpl, cv2.TM_CCOEFF_NORMED)
        _, corr_score, _, corr_loc = cv2.minMaxLoc(corr_map)

        sqdiff_map = cv2.matchTemplate(frame_blur, tmpl, cv2.TM_SQDIFF_NORMED)
        sq_min, _, sq_loc, _ = cv2.minMaxLoc(sqdiff_map)
        sq_score = 1.0 - sq_min

        if corr_score >= sq_score:
            score = float(corr_score)
            top_left = corr_loc
        else:
            score = float(sq_score)
            top_left = sq_loc

        x, y = top_left
        center_x = x + (tmpl_w / 2.0)
        center_y = y + (tmpl_h / 2.0)
        dx = abs(center_x - (frame_w / 2.0)) / max(frame_w / 2.0, 1.0)
        dy = abs(center_y - (frame_h / 2.0)) / max(frame_h / 2.0, 1.0)
        center_distance = min(1.0, float(np.hypot(dx, dy)))
        rank_score = score * (0.55 + 0.45 * template["scale_ratio"]) * (1.0 - 0.25 * center_distance)

        if best_match is None or rank_score > best_match["rank_score"]:
            best_match = {
                "found": True,
                "score": score,
                "rank_score": rank_score,
                "scale": template["scale"],
                "box": (x, y, tmpl_w, tmpl_h),
            }

    if best_match is None or best_match["score"] < REFERENCE_MATCH_THRESHOLD:
        return {
            "found": False,
            "score": 0.0 if best_match is None else best_match["score"],
            "box": None,
        }

    x, y, w, h = best_match["box"]
    best_match["roi"] = frame[y : y + h, x : x + w]
    return best_match


def build_default_results():
    return {
        "cnn": {"score": 1.0, "passed": True},
        "ssim": {"score": 1.0, "passed": True},
        "perceptual": {"score": 0.0, "passed": True},
        "contour": {"score": 0, "passed": True},
    }


def build_missing_results():
    return {
        "cnn": {"score": 0.0, "passed": False},
        "ssim": {"score": 0.0, "passed": False},
        "perceptual": {"score": 1.0, "passed": False},
        "contour": {"score": CONTOUR_DIFF_THRESHOLD + 1, "passed": False},
    }


def summarize_status(results, match_info):
    failed = sum(1 for r in results.values() if not r["passed"])

    if not match_info["found"]:
        return {
            "level": "missing",
            "text": "REFERENCE NOT IN VIEW",
            "failed": len(results),
            "match_score": match_info["score"],
            "box": None,
        }

    if failed >= WARNING_THRESHOLD:
        level = "defect"
        text = f"WARNING!! DEFECT DETECTED ({failed}/4 failed)"
    elif failed == 0:
        level = "ok"
        text = "OK - Pattern matches reference"
    else:
        level = "ok"
        text = f"OK - Minor drift ({failed}/4 failed)"

    return {
        "level": level,
        "text": text,
        "failed": failed,
        "match_score": match_info["score"],
        "box": match_info.get("box"),
    }


def compute_frame_scores(extractor, ref_features, ref_layers, ref_gray, frame):
    """Compute raw comparison scores without deciding pass/fail yet."""
    test_features = extractor.extract_features_from_frame(frame)
    cnn_sim = extractor.cosine_similarity(ref_features, test_features)

    test_gray_native = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.resize(test_gray_native, (ref_gray.shape[1], ref_gray.shape[0]))
    ssim_score, ssim_diff = ssim(ref_gray, test_gray, full=True)

    test_layers = extractor.extract_layer_features_from_frame(frame)
    perc_diff = extractor.perceptual_difference(ref_layers, test_layers)

    contour_ref_gray = cv2.resize(
        ref_gray,
        (test_gray_native.shape[1], test_gray_native.shape[0]),
        interpolation=cv2.INTER_AREA,
    )
    ref_blur = cv2.GaussianBlur(contour_ref_gray, (5, 5), 1.5)
    test_blur = cv2.GaussianBlur(test_gray_native, (5, 5), 1.5)
    ref_edges = cv2.Canny(ref_blur, 50, 150)
    test_edges = cv2.Canny(test_blur, 50, 150)
    diff = cv2.absdiff(ref_edges, test_edges)
    diff_dilated = cv2.dilate(diff, None, iterations=3)
    diff_contours, _ = cv2.findContours(
        diff_dilated,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    sig_diffs = len([c for c in diff_contours if cv2.contourArea(c) > MIN_CONTOUR_AREA])

    ssim_diff_map = (ssim_diff * 255).astype(np.uint8)
    scores = {
        "cnn": float(cnn_sim),
        "ssim": float(ssim_score),
        "perceptual": float(perc_diff),
        "contour": int(sig_diffs),
    }
    return scores, ssim_diff_map


def apply_thresholds(scores, thresholds):
    """Convert raw scores into pass/fail results."""
    return {
        "cnn": {
            "score": scores["cnn"],
            "passed": scores["cnn"] >= thresholds["cnn"],
        },
        "ssim": {
            "score": scores["ssim"],
            "passed": scores["ssim"] >= thresholds["ssim"],
        },
        "perceptual": {
            "score": scores["perceptual"],
            "passed": scores["perceptual"] <= thresholds["perceptual"],
        },
        "contour": {
            "score": scores["contour"],
            "passed": scores["contour"] <= thresholds["contour"],
        },
    }


def build_adaptive_thresholds(extractor, reference_state, calibration_rois):
    """
    Learn camera-session thresholds from the calibration frames themselves.
    This keeps the system strict for defects while tolerating the real camera noise.
    """
    thresholds = default_thresholds()
    if not calibration_rois:
        return thresholds

    score_samples = {key: [] for key in thresholds}
    for roi in calibration_rois:
        analysis_roi = extract_core_pattern(roi)
        scores, _ = compute_frame_scores(
            extractor,
            reference_state["features"],
            reference_state["layers"],
            reference_state["gray"],
            analysis_roi,
        )
        for key, value in scores.items():
            score_samples[key].append(float(value))

    if len(score_samples["cnn"]) < 3:
        return thresholds

    thresholds["cnn"] = min(
        CNN_SIMILARITY_THRESHOLD,
        max(0.55, min(score_samples["cnn"]) - 0.03),
    )
    thresholds["ssim"] = min(
        SSIM_THRESHOLD,
        max(0.10, min(score_samples["ssim"]) - 0.05),
    )
    thresholds["perceptual"] = max(
        PERCEPTUAL_DIFF_THRESHOLD,
        min(0.80, max(score_samples["perceptual"]) + 0.06),
    )
    thresholds["contour"] = max(
        CONTOUR_DIFF_THRESHOLD,
        min(12, int(max(score_samples["contour"])) + 1),
    )
    return thresholds


def analyze_frame(extractor, ref_features, ref_layers, ref_gray, frame, thresholds):
    """Analyze a localized camera ROI against the reference."""
    scores, ssim_diff_map = compute_frame_scores(
        extractor,
        ref_features,
        ref_layers,
        ref_gray,
        frame,
    )
    return apply_thresholds(scores, thresholds), ssim_diff_map


def draw_overlay(frame, results, ref_img, fps, frame_count, status):
    """Draw status overlay on the camera frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    bar_h = 80
    if status["level"] in {"missing", "searching", "calibrating"}:
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 140, 255), -1)
    elif status["level"] == "defect":
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 180), -1)
    else:
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 130, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, overlay)

    cv2.putText(
        overlay,
        status["text"],
        (15, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
    )

    if status["level"] == "missing":
        scores_text = f"Match:{status['match_score']:.3f}  Move the product fully into view"
    elif status["level"] == "calibrating":
        scores_text = (
            f"Match:{status['match_score']:.3f}  "
            f"Keep a GOOD sample steady in view"
        )
    elif status["level"] == "searching":
        scores_text = "Waiting for first inspection..."
    else:
        cnn = results["cnn"]["score"]
        ssim_score = results["ssim"]["score"]
        perc = results["perceptual"]["score"]
        cont = results["contour"]["score"]
        scores_text = (
            f"Match:{status['match_score']:.3f}  CNN:{cnn:.3f}  "
            f"SSIM:{ssim_score:.1%}  Perc:{perc:.4f}  Cont:{cont}"
        )
    cv2.putText(
        overlay,
        scores_text,
        (15, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
    )

    if status["box"] is not None:
        x, y, box_w, box_h = status["box"]
        cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 255, 255), 2)
        cv2.putText(
            overlay,
            "ROI",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            2,
        )

    ref_small = cv2.resize(ref_img, (120, 90))
    overlay[bar_h + 10 : bar_h + 100, w - 130 : w - 10] = ref_small
    cv2.rectangle(overlay, (w - 132, bar_h + 8), (w - 8, bar_h + 102), (255, 255, 255), 2)
    cv2.putText(
        overlay,
        "REF",
        (w - 120, bar_h + 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 255, 255),
        1,
    )

    info_y = h - 30
    cv2.rectangle(overlay, (0, h - 40), (w, h), (30, 30, 30), -1)
    timestamp = datetime.now().strftime("%H:%M:%S")
    info_text = f"[{timestamp}]  Frame: {frame_count}  |  FPS: {fps:.1f}"
    cv2.putText(
        overlay,
        info_text,
        (10, info_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (180, 180, 180),
        1,
    )

    return overlay


def camera_loop(ref_path, camera_id=0, check_interval=2.0):
    print()
    print("=" * 60)
    print("  VISION MACHINE CAMERA - Live Monitoring")
    print("  Powered by ResNet50 CNN + OpenCV")
    print("=" * 60)
    print()

    print(f"  Reference: {ref_path}")
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print(f"  ERROR: Cannot load reference image: {ref_path}")
        sys.exit(1)

    print("  Loading CNN model...")
    extractor = CNNFeatureExtractor()
    print("  Model ready!")

    print(f"  Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open camera {camera_id}")
        print("  Try a different camera ID with --camera 1")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera resolution: {actual_w}x{actual_h}")

    active_reference = build_reference_state(extractor, ref_img, (actual_h, actual_w))
    if not active_reference["templates"]:
        print("  ERROR: Reference image cannot be searched at this camera resolution.")
        cap.release()
        sys.exit(1)
    print(f"  Prepared {len(active_reference['templates'])} reference search scale(s).")
    if LIVE_BASELINE_FRAMES > 0:
        print(
            "  Live calibration enabled: hold a GOOD sample steady "
            f"for the first {LIVE_BASELINE_FRAMES} detections."
        )
    print()
    print("  Starting live monitoring...")
    print("  Close the window or press Ctrl+C to stop.")
    print()

    frame_count = 0
    last_check_time = 0.0
    fps = 0.0
    fps_timer = time.time()
    fps_count = 0
    total_defects = 0
    total_missing = 0
    live_reference_ready = LIVE_BASELINE_FRAMES <= 0
    calibration_rois = []

    current_results = build_default_results()
    current_status = {
        "level": "searching",
        "text": "Searching for reference...",
        "failed": 0,
        "match_score": 0.0,
        "box": None,
    }

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("  Camera read failed!")
                break

            frame_count += 1
            fps_count += 1
            current_time = time.time()

            if current_time - fps_timer >= 1.0:
                fps = fps_count / (current_time - fps_timer)
                fps_count = 0
                fps_timer = current_time

            if (current_time - last_check_time) >= check_interval:
                last_check_time = current_time
                timestamp = datetime.now().strftime("%H:%M:%S")
                match_info = locate_reference_region(
                    frame,
                    active_reference["templates"],
                    active_reference["gray"].shape,
                )

                if not match_info["found"]:
                    total_missing += 1
                    current_results = build_missing_results()
                    current_status = summarize_status(current_results, match_info)
                    print(
                        f"  [{timestamp}] Frame #{frame_count:05d} | "
                        f"REFERENCE NOT IN VIEW | Match:{match_info['score']:.3f}"
                    )
                elif not live_reference_ready:
                    if match_info["score"] >= LIVE_REFERENCE_MIN_MATCH:
                        calibration_rois.append(match_info["roi"])
                    calibration_count = min(len(calibration_rois), LIVE_BASELINE_FRAMES)
                    current_results = build_default_results()
                    current_status = {
                        "level": "calibrating",
                        "text": f"CALIBRATING LIVE REFERENCE ({calibration_count}/{LIVE_BASELINE_FRAMES})",
                        "failed": 0,
                        "match_score": match_info["score"],
                        "box": match_info.get("box"),
                    }
                    print(
                        f"  [{timestamp}] Frame #{frame_count:05d} | "
                        f"CALIBRATING ({calibration_count}/{LIVE_BASELINE_FRAMES}) | "
                        f"Match:{match_info['score']:.3f}"
                    )

                    if len(calibration_rois) >= LIVE_BASELINE_FRAMES:
                        live_ref_img = build_live_reference(calibration_rois)
                        active_reference = build_reference_state(
                            extractor,
                            live_ref_img,
                            (actual_h, actual_w),
                        )
                        active_reference["thresholds"] = build_adaptive_thresholds(
                            extractor,
                            active_reference,
                            calibration_rois,
                        )
                        live_reference_ready = True
                        current_status = {
                            "level": "searching",
                            "text": "Live reference ready",
                            "failed": 0,
                            "match_score": match_info["score"],
                            "box": match_info.get("box"),
                        }
                        thresholds = active_reference["thresholds"]
                        print(
                            "  Adaptive thresholds | "
                            f"CNN>={thresholds['cnn']:.3f} "
                            f"SSIM>={thresholds['ssim']:.3f} "
                            f"Perc<={thresholds['perceptual']:.3f} "
                            f"Cont<={thresholds['contour']}"
                        )
                        print("  Live camera reference captured. Starting defect detection.")
                else:
                    analysis_roi = extract_core_pattern(match_info["roi"])
                    current_results, _ = analyze_frame(
                        extractor,
                        active_reference["features"],
                        active_reference["layers"],
                        active_reference["gray"],
                        analysis_roi,
                        active_reference["thresholds"],
                    )
                    current_status = summarize_status(current_results, match_info)

                    failed = current_status["failed"]
                    cnn = current_results["cnn"]["score"]
                    ssim_score = current_results["ssim"]["score"]
                    perc = current_results["perceptual"]["score"]

                    if current_status["level"] == "defect":
                        total_defects += 1
                        print(
                            f"  [{timestamp}] Frame #{frame_count:05d} | "
                            f"WARNING!! DEFECT ({failed}/4 failed) | "
                            f"Match:{match_info['score']:.3f} CNN:{cnn:.3f} "
                            f"SSIM:{ssim_score:.1%} Perc:{perc:.4f}"
                        )
                    elif failed == 0:
                        print(
                            f"  [{timestamp}] Frame #{frame_count:05d} | "
                            f"OK | Match:{match_info['score']:.3f} CNN:{cnn:.3f} "
                            f"SSIM:{ssim_score:.1%} Perc:{perc:.4f}"
                        )
                    else:
                        print(
                            f"  [{timestamp}] Frame #{frame_count:05d} | "
                            f"OK ({failed}/4 minor) | Match:{match_info['score']:.3f} "
                            f"CNN:{cnn:.3f} SSIM:{ssim_score:.1%} Perc:{perc:.4f}"
                        )

            display = draw_overlay(
                frame,
                current_results,
                active_reference["image"],
                fps,
                frame_count,
                current_status,
            )
            cv2.imshow("Vision Machine Camera - Live Monitor", display)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            if (
                cv2.getWindowProperty("Vision Machine Camera - Live Monitor", cv2.WND_PROP_VISIBLE)
                < 1
            ):
                break

    except KeyboardInterrupt:
        pass

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("  " + "=" * 50)
    print("  SESSION COMPLETE")
    print(f"  Total frames: {frame_count}")
    print(f"  Defects detected: {total_defects}")
    print(f"  Reference missing: {total_missing}")
    print("  " + "=" * 50)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Vision Machine Camera - Live Defect Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python camera.py -r images/reference_straight.png
    python camera.py -r images/reference_straight.png --camera 1
    python camera.py -r images/reference_straight.png --interval 3

Close the window or press Ctrl+C to stop.
        """,
    )
    parser.add_argument(
        "-r",
        "--reference",
        required=True,
        help="Path to the reference (good) image",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        help="Camera device ID (default: 0)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Seconds between each CNN check (default: 2.0)",
    )

    args = parser.parse_args()
    camera_loop(args.reference, args.camera, args.interval)


if __name__ == "__main__":
    main()
