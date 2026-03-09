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
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from datetime import datetime
import io


# ═══════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════

CNN_SIMILARITY_THRESHOLD = 0.90
SSIM_THRESHOLD = 0.85
PERCEPTUAL_DIFF_THRESHOLD = 0.15
CONTOUR_DIFF_THRESHOLD = 3
MIN_CONTOUR_AREA = 1500
CNN_INPUT_SIZE = (224, 224)
WARNING_THRESHOLD = 2


# ═══════════════════════════════════════════
#  CNN FEATURE EXTRACTOR
# ═══════════════════════════════════════════

class CNNFeatureExtractor:
    def __init__(self):
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model.eval()
        self.feature_layers = torch.nn.Sequential(*list(self.model.children())[:-1])

        children = list(self.model.children())
        self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        self.layer_extractors = {}
        for i, name in enumerate(self.layer_names):
            self.layer_extractors[name] = torch.nn.Sequential(*children[:4 + i + 1])

        self.preprocess = transforms.Compose([
            transforms.Resize(CNN_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def extract_features_from_frame(self, frame):
        """Extract features from an OpenCV frame (BGR numpy array)."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features = self.feature_layers(tensor)
        return features.flatten()

    def extract_layer_features_from_frame(self, frame):
        """Extract multi-layer features from an OpenCV frame."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = self.preprocess(img).unsqueeze(0)
        layer_features = {}
        with torch.no_grad():
            for name, extractor in self.layer_extractors.items():
                layer_features[name] = extractor(tensor)
        return layer_features

    def extract_features_from_path(self, image_path):
        """Extract features from an image file."""
        img = Image.open(image_path).convert('RGB')
        tensor = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features = self.feature_layers(tensor)
        return features.flatten()

    def extract_layer_features_from_path(self, image_path):
        """Extract multi-layer features from an image file."""
        img = Image.open(image_path).convert('RGB')
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
        weights = {'layer1': 0.1, 'layer2': 0.2, 'layer3': 0.3, 'layer4': 0.4}
        for name in self.layer_names:
            f1 = layers1[name].flatten()
            f2 = layers2[name].flatten()
            cos_sim = F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
            total_diff += weights[name] * (1.0 - cos_sim)
        return total_diff


# ═══════════════════════════════════════════
#  FRAME ANALYSIS
# ═══════════════════════════════════════════

def analyze_frame(extractor, ref_features, ref_layers, ref_gray, frame):
    """Analyze a single camera frame against the reference."""
    results = {}

    # Method 1: CNN Deep Features
    test_features = extractor.extract_features_from_frame(frame)
    cnn_sim = extractor.cosine_similarity(ref_features, test_features)
    results['cnn'] = {'score': cnn_sim, 'passed': cnn_sim >= CNN_SIMILARITY_THRESHOLD}

    # Method 2: SSIM
    test_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    test_gray = cv2.resize(test_gray, (ref_gray.shape[1], ref_gray.shape[0]))
    ssim_score, ssim_diff = ssim(ref_gray, test_gray, full=True)
    results['ssim'] = {'score': ssim_score, 'passed': ssim_score >= SSIM_THRESHOLD}

    # Method 3: Perceptual Difference
    test_layers = extractor.extract_layer_features_from_frame(frame)
    perc_diff = extractor.perceptual_difference(ref_layers, test_layers)
    results['perceptual'] = {'score': perc_diff, 'passed': perc_diff <= PERCEPTUAL_DIFF_THRESHOLD}

    # Method 4: Contour Analysis
    ref_blur = cv2.GaussianBlur(ref_gray, (5, 5), 1.5)
    test_blur = cv2.GaussianBlur(test_gray, (5, 5), 1.5)
    ref_edges = cv2.Canny(ref_blur, 50, 150)
    test_edges = cv2.Canny(test_blur, 50, 150)
    diff = cv2.absdiff(ref_edges, test_edges)
    diff_dilated = cv2.dilate(diff, None, iterations=3)
    diff_contours, _ = cv2.findContours(diff_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sig_diffs = len([c for c in diff_contours if cv2.contourArea(c) > MIN_CONTOUR_AREA])
    results['contour'] = {'score': sig_diffs, 'passed': sig_diffs <= CONTOUR_DIFF_THRESHOLD}

    # SSIM diff map for overlay
    ssim_diff_map = (ssim_diff * 255).astype(np.uint8)

    return results, ssim_diff_map


def draw_overlay(frame, results, ref_img, fps, frame_count):
    """Draw status overlay on the camera frame."""
    h, w = frame.shape[:2]
    overlay = frame.copy()

    failed = sum(1 for r in results.values() if not r['passed'])
    is_defect = failed >= WARNING_THRESHOLD

    # Draw status bar at top
    bar_h = 80
    if is_defect:
        # Red warning bar
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 180), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, overlay)
        status_text = f"WARNING!! DEFECT DETECTED ({failed}/4 failed)"
        cv2.putText(overlay, status_text, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    else:
        # Green OK bar
        cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 130, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, overlay)
        status_text = "OK - Pattern matches reference"
        cv2.putText(overlay, status_text, (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Draw scores
    cnn = results['cnn']['score']
    ssim_s = results['ssim']['score']
    perc = results['perceptual']['score']
    cont = results['contour']['score']
    scores_text = f"CNN:{cnn:.3f}  SSIM:{ssim_s:.1%}  Perc:{perc:.4f}  Cont:{cont}"
    cv2.putText(overlay, scores_text, (15, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    # Draw reference thumbnail in corner
    ref_small = cv2.resize(ref_img, (120, 90))
    overlay[bar_h + 10:bar_h + 100, w - 130:w - 10] = ref_small
    cv2.rectangle(overlay, (w - 132, bar_h + 8), (w - 8, bar_h + 102), (255, 255, 255), 2)
    cv2.putText(overlay, "REF", (w - 120, bar_h + 115),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Draw bottom info bar
    info_y = h - 30
    cv2.rectangle(overlay, (0, h - 40), (w, h), (30, 30, 30), -1)
    timestamp = datetime.now().strftime("%H:%M:%S")
    info_text = f"[{timestamp}]  Frame: {frame_count}  |  FPS: {fps:.1f}"
    cv2.putText(overlay, info_text, (10, info_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return overlay


# ═══════════════════════════════════════════
#  MAIN CAMERA LOOP
# ═══════════════════════════════════════════

def camera_loop(ref_path, camera_id=0, check_interval=2.0):
    """
    Main camera monitoring loop.
    - Captures from webcam
    - Compares every `check_interval` seconds against reference
    - Shows live feed with overlay
    """
    print()
    print("=" * 60)
    print("  VISION MACHINE CAMERA - Live Monitoring")
    print("  Powered by ResNet50 CNN + OpenCV")
    print("=" * 60)
    print()

    # Load reference
    print(f"  Reference: {ref_path}")
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print(f"  ERROR: Cannot load reference image: {ref_path}")
        sys.exit(1)

    ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

    # Load CNN
    print("  Loading CNN model...")
    extractor = CNNFeatureExtractor()
    print("  Computing reference features...")
    ref_features = extractor.extract_features_from_path(ref_path)
    ref_layers = extractor.extract_layer_features_from_path(ref_path)
    print("  Model ready!")

    # Open camera
    print(f"  Opening camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open camera {camera_id}")
        print("  Try a different camera ID with --camera 1")
        sys.exit(1)

    # Set camera resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Camera resolution: {actual_w}x{actual_h}")
    print()
    print("  Starting live monitoring...")
    print("  Close the window or press Ctrl+C to stop.")
    print()

    # State
    frame_count = 0
    last_check_time = 0
    fps = 0.0
    fps_timer = time.time()
    fps_count = 0
    total_defects = 0

    # Default results (before first check)
    current_results = {
        'cnn': {'score': 1.0, 'passed': True},
        'ssim': {'score': 1.0, 'passed': True},
        'perceptual': {'score': 0.0, 'passed': True},
        'contour': {'score': 0, 'passed': True},
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

            # Calculate FPS
            if current_time - fps_timer >= 1.0:
                fps = fps_count / (current_time - fps_timer)
                fps_count = 0
                fps_timer = current_time

            # Run detection at interval (not every frame - too slow)
            if (current_time - last_check_time) >= check_interval:
                last_check_time = current_time
                current_results, _ = analyze_frame(
                    extractor, ref_features, ref_layers, ref_gray, frame
                )

                # Console output
                failed = sum(1 for r in current_results.values() if not r['passed'])
                is_defect = failed >= WARNING_THRESHOLD
                timestamp = datetime.now().strftime("%H:%M:%S")
                cnn = current_results['cnn']['score']
                ssim_s = current_results['ssim']['score']
                perc = current_results['perceptual']['score']

                if is_defect:
                    total_defects += 1
                    print(f"  [{timestamp}] Frame #{frame_count:05d} | WARNING!! DEFECT ({failed}/4 failed) | CNN:{cnn:.3f} SSIM:{ssim_s:.1%} Perc:{perc:.4f}")
                else:
                    print(f"  [{timestamp}] Frame #{frame_count:05d} | OK | CNN:{cnn:.3f} SSIM:{ssim_s:.1%} Perc:{perc:.4f}")

            # Draw overlay on frame
            display = draw_overlay(frame, current_results, ref_img, fps, frame_count)

            # Show window
            cv2.imshow("Vision Machine Camera - Live Monitor", display)

            # Allow window to process events (close button)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC as fallback
                break

            # Check if window was closed
            if cv2.getWindowProperty("Vision Machine Camera - Live Monitor", cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        pass

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    print()
    print("  " + "=" * 50)
    print(f"  SESSION COMPLETE")
    print(f"  Total frames: {frame_count}")
    print(f"  Defects detected: {total_defects}")
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
        """
    )
    parser.add_argument('-r', '--reference', required=True,
                        help='Path to the reference (good) image')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (default: 0)')
    parser.add_argument('--interval', type=float, default=2.0,
                        help='Seconds between each CNN check (default: 2.0)')

    args = parser.parse_args()
    camera_loop(args.reference, args.camera, args.interval)


if __name__ == "__main__":
    main()
