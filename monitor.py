"""
Vision Machine Monitor - Continuous Loop Mode
==============================================
Monitors a folder of images in sequence, comparing each against a reference.
Simulates a live camera feed by processing images one-by-one with status output.

Usage:
    python monitor.py --reference images/reference_straight.png --folder images/
    python monitor.py --reference images/reference_straight.png --images img1.png img2.png img3.png
    python monitor.py --reference images/reference_straight.png --folder images/ --delay 2
"""

import argparse
import os
import sys
import time
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from datetime import datetime


# ═══════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════

CNN_SIMILARITY_THRESHOLD = 0.90
SSIM_THRESHOLD = 0.85
PERCEPTUAL_DIFF_THRESHOLD = 0.15
CONTOUR_DIFF_THRESHOLD = 3
MIN_CONTOUR_AREA = 1500
CNN_INPUT_SIZE = (224, 224)

# How many methods must fail to trigger WARNING
WARNING_THRESHOLD = 2  # If 2+ methods fail → WARNING


# ═══════════════════════════════════════════
#  CNN FEATURE EXTRACTOR (shared instance)
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

    def extract_features(self, image_path):
        img = Image.open(image_path).convert('RGB')
        tensor = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features = self.feature_layers(tensor)
        return features.flatten()

    def extract_layer_features(self, image_path):
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
#  QUICK ANALYSIS (optimized for loop mode)
# ═══════════════════════════════════════════

def quick_analyze(extractor, ref_features, ref_layers, ref_path, test_path):
    """
    Run all 4 methods and return a summary dict.
    Uses pre-computed reference features for speed.
    """
    results = {}

    # Method 1: CNN Deep Features
    test_features = extractor.extract_features(test_path)
    cnn_sim = extractor.cosine_similarity(ref_features, test_features)
    results['cnn'] = {'score': cnn_sim, 'passed': cnn_sim >= CNN_SIMILARITY_THRESHOLD}

    # Method 2: SSIM
    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    test_img = cv2.resize(test_img, (ref_img.shape[1], ref_img.shape[0]))
    ssim_score, _ = ssim(ref_img, test_img, full=True)
    results['ssim'] = {'score': ssim_score, 'passed': ssim_score >= SSIM_THRESHOLD}

    # Method 3: Perceptual Difference
    test_layers = extractor.extract_layer_features(test_path)
    perc_diff = extractor.perceptual_difference(ref_layers, test_layers)
    results['perceptual'] = {'score': perc_diff, 'passed': perc_diff <= PERCEPTUAL_DIFF_THRESHOLD}

    # Method 4: Contour Analysis
    ref_blur = cv2.GaussianBlur(ref_img, (5, 5), 1.5)
    test_blur = cv2.GaussianBlur(test_img, (5, 5), 1.5)
    ref_edges = cv2.Canny(ref_blur, 50, 150)
    test_edges = cv2.Canny(test_blur, 50, 150)
    diff = cv2.absdiff(ref_edges, test_edges)
    diff_dilated = cv2.dilate(diff, None, iterations=3)
    diff_contours, _ = cv2.findContours(diff_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sig_diffs = len([c for c in diff_contours if cv2.contourArea(c) > MIN_CONTOUR_AREA])
    results['contour'] = {'score': sig_diffs, 'passed': sig_diffs <= CONTOUR_DIFF_THRESHOLD}

    return results


def print_loop_status(image_name, results, frame_num):
    """Print a single line status for the current frame in the loop."""
    failed = sum(1 for r in results.values() if not r['passed'])
    is_defect = failed >= WARNING_THRESHOLD
    timestamp = datetime.now().strftime("%H:%M:%S")

    cnn_score = results['cnn']['score']
    ssim_score = results['ssim']['score']
    perc_score = results['perceptual']['score']
    contour_score = results['contour']['score']

    if is_defect:
        # ⚠️ WARNING output
        print(f"  [{timestamp}] Frame #{frame_num:03d} | {image_name}")
        print(f"  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"  !!  WARNING!!  DEFECT DETECTED  -  {failed}/4 methods failed  !!")
        print(f"  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"  !!  CNN: {cnn_score:.3f}  |  SSIM: {ssim_score:.1%}  |  Perceptual: {perc_score:.4f}  |  Contours: {contour_score}")

        # Show which methods failed
        failed_names = []
        if not results['cnn']['passed']:
            failed_names.append("CNN")
        if not results['ssim']['passed']:
            failed_names.append("SSIM")
        if not results['perceptual']['passed']:
            failed_names.append("Perceptual")
        if not results['contour']['passed']:
            failed_names.append("Contour")
        print(f"  !!  Failed: {', '.join(failed_names)}")
        print(f"  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print()
    else:
        # ✅ OK output
        status_icon = "OK" if failed == 0 else f"OK ({failed}/4 minor)"
        print(f"  [{timestamp}] Frame #{frame_num:03d} | {image_name:30s} | {status_icon} | CNN:{cnn_score:.3f} SSIM:{ssim_score:.1%} Perc:{perc_score:.4f} Cont:{contour_score}")


# ═══════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════

def monitor_loop(ref_path, image_list, delay=1.0, loop_forever=False):
    """
    Run defect detection in a loop over a list of images.
    Simulates continuous camera monitoring.
    """
    print()
    print("=" * 70)
    print("  VISION MACHINE MONITOR - Continuous Monitoring Mode")
    print("  Powered by ResNet50 CNN + OpenCV")
    print("=" * 70)
    print()
    print(f"  Reference: {os.path.basename(ref_path)}")
    print(f"  Images to check: {len(image_list)}")
    print(f"  Delay between frames: {delay}s")
    if loop_forever:
        print(f"  Mode: CONTINUOUS (Ctrl+C to stop)")
    else:
        print(f"  Mode: SINGLE PASS")
    print()
    print("  Loading CNN model...")
    extractor = CNNFeatureExtractor()

    # Pre-compute reference features (done once for speed)
    print("  Computing reference features...")
    ref_features = extractor.extract_features(ref_path)
    ref_layers = extractor.extract_layer_features(ref_path)
    print("  Ready! Starting monitoring...\n")

    print("  " + "-" * 66)
    print(f"  {'TIME':10s} {'FRAME':10s} | {'IMAGE':30s} | {'STATUS'}")
    print("  " + "-" * 66)

    frame_num = 0
    total_defects = 0

    try:
        while True:
            for img_path in image_list:
                frame_num += 1

                # Skip the reference image itself
                if os.path.abspath(img_path) == os.path.abspath(ref_path):
                    continue

                results = quick_analyze(extractor, ref_features, ref_layers, ref_path, img_path)
                print_loop_status(os.path.basename(img_path), results, frame_num)

                failed = sum(1 for r in results.values() if not r['passed'])
                if failed >= WARNING_THRESHOLD:
                    total_defects += 1

                time.sleep(delay)

            if not loop_forever:
                break

    except KeyboardInterrupt:
        print("\n\n  Monitoring stopped by user.")

    # Summary
    print()
    print("  " + "=" * 66)
    print(f"  MONITORING COMPLETE")
    print(f"  Total frames checked: {frame_num}")
    print(f"  Defects detected: {total_defects}")
    if total_defects > 0:
        print(f"  WARNING!! {total_defects} defective frame(s) found!")
    else:
        print(f"  All frames passed inspection.")
    print("  " + "=" * 66)
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Vision Machine Monitor - Continuous Loop Defect Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Monitor all images in a folder
  python monitor.py -r images/reference_straight.png -f images/

  # Monitor specific images
  python monitor.py -r images/reference_straight.png -i img1.png img2.png img3.png

  # Continuous loop with 2 second delay
  python monitor.py -r images/reference_straight.png -f images/ --delay 2 --loop
        """
    )
    parser.add_argument('-r', '--reference', required=True,
                        help='Path to the reference (good) image')
    parser.add_argument('-f', '--folder',
                        help='Folder containing test images to monitor')
    parser.add_argument('-i', '--images', nargs='+',
                        help='Specific image paths to check')
    parser.add_argument('--delay', type=float, default=1.0,
                        help='Seconds between each frame check (default: 1.0)')
    parser.add_argument('--loop', action='store_true',
                        help='Loop forever (Ctrl+C to stop)')

    args = parser.parse_args()

    if not os.path.exists(args.reference):
        print(f"  Reference image not found: {args.reference}")
        sys.exit(1)

    # Collect image list
    image_list = []

    if args.images:
        image_list = args.images
    elif args.folder:
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tiff']
        for ext in extensions:
            image_list.extend(glob.glob(os.path.join(args.folder, ext)))
        image_list.sort()
    else:
        print("  Please specify --folder or --images")
        sys.exit(1)

    if not image_list:
        print("  No images found!")
        sys.exit(1)

    monitor_loop(args.reference, image_list, args.delay, args.loop)


if __name__ == "__main__":
    main()
