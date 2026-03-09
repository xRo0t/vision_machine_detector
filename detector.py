"""
Vision Machine Detector - Factory Defect Detection
===================================================
Compares a reference image against a test image using multiple methods:
  1. CNN Deep Features (ResNet50) - Cosine similarity of deep features
  2. SSIM (Structural Similarity) - Structural comparison
  3. Perceptual Diff (CNN-based) - Layer-by-layer perceptual difference
  4. Contour Analysis (OpenCV) - Shape/edge comparison

Usage:
    python detector.py --reference images/reference.jpg --test images/test.jpg
    python detector.py --reference images/reference.jpg --test images/test.jpg --save-diff
"""

import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from skimage.metrics import structural_similarity as ssim
from PIL import Image


# ═══════════════════════════════════════════
#  CONFIGURATION - Tune these for your factory
# ═══════════════════════════════════════════

# CNN Deep Feature similarity threshold (0-1, higher = stricter)
CNN_SIMILARITY_THRESHOLD = 0.90

# SSIM threshold (0-1, higher = stricter)
SSIM_THRESHOLD = 0.85

# Perceptual difference threshold (0-1, lower = stricter)
PERCEPTUAL_DIFF_THRESHOLD = 0.15

# Contour difference threshold (number of significantly different contours)
CONTOUR_DIFF_THRESHOLD = 3

# Minimum contour area to consider (filters noise)
MIN_CONTOUR_AREA = 1500

# Image resize for CNN (don't change unless you know what you're doing)
CNN_INPUT_SIZE = (224, 224)


# ═══════════════════════════════════════════
#  CNN FEATURE EXTRACTOR
# ═══════════════════════════════════════════

class CNNFeatureExtractor:
    """
    Uses a pre-trained ResNet50 to extract deep features from images.
    These features capture high-level patterns, shapes, and textures -
    much smarter than pixel-level comparison.
    """

    def __init__(self):
        print("  🧠 Loading ResNet50 model...")
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model.eval()

        # Remove the final classification layer - we want features, not classes
        self.feature_layers = torch.nn.Sequential(*list(self.model.children())[:-1])

        # Also keep intermediate layers for perceptual comparison
        self.layer_extractors = {}
        children = list(self.model.children())
        self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        for i, name in enumerate(self.layer_names):
            self.layer_extractors[name] = torch.nn.Sequential(*children[:4 + i + 1])

        self.preprocess = transforms.Compose([
            transforms.Resize(CNN_INPUT_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        print("  ✅ Model loaded successfully!")

    def extract_features(self, image_path):
        """Extract the final feature vector from an image."""
        img = Image.open(image_path).convert('RGB')
        tensor = self.preprocess(img).unsqueeze(0)
        with torch.no_grad():
            features = self.feature_layers(tensor)
        return features.flatten()

    def extract_layer_features(self, image_path):
        """Extract features from multiple layers for perceptual comparison."""
        img = Image.open(image_path).convert('RGB')
        tensor = self.preprocess(img).unsqueeze(0)
        layer_features = {}
        with torch.no_grad():
            for name, extractor in self.layer_extractors.items():
                layer_features[name] = extractor(tensor)
        return layer_features

    def cosine_similarity(self, feat1, feat2):
        """Compute cosine similarity between two feature vectors."""
        return F.cosine_similarity(feat1.unsqueeze(0), feat2.unsqueeze(0)).item()

    def perceptual_difference(self, layers1, layers2):
        """
        Compute perceptual difference across multiple CNN layers.
        This captures differences at multiple scales:
        - Early layers: edges, textures
        - Middle layers: patterns, shapes
        - Deep layers: high-level structure
        """
        total_diff = 0.0
        weights = {'layer1': 0.1, 'layer2': 0.2, 'layer3': 0.3, 'layer4': 0.4}

        for name in self.layer_names:
            f1 = layers1[name].flatten()
            f2 = layers2[name].flatten()
            # Cosine distance (1 - cosine_similarity) for each layer
            cos_sim = F.cosine_similarity(f1.unsqueeze(0), f2.unsqueeze(0)).item()
            layer_diff = 1.0 - cos_sim
            total_diff += weights[name] * layer_diff

        return total_diff


# ═══════════════════════════════════════════
#  OPENCV ANALYSIS
# ═══════════════════════════════════════════

def compute_ssim(ref_path, test_path):
    """Compute Structural Similarity Index between two images."""
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)

    # Resize test to match reference
    test = cv2.resize(test, (ref.shape[1], ref.shape[0]))

    score, diff_map = ssim(ref, test, full=True)
    diff_map = (diff_map * 255).astype(np.uint8)

    return score, diff_map


def analyze_contours(ref_path, test_path):
    """
    Compare contours between reference and test images.
    Returns number of significant differences and a visualization.
    """
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    test = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    test = cv2.resize(test, (ref.shape[1], ref.shape[0]))

    # Apply Gaussian blur to reduce noise before edge detection
    ref_blur = cv2.GaussianBlur(ref, (5, 5), 1.5)
    test_blur = cv2.GaussianBlur(test, (5, 5), 1.5)

    # Edge detection
    ref_edges = cv2.Canny(ref_blur, 50, 150)
    test_edges = cv2.Canny(test_blur, 50, 150)

    # Find contours
    ref_contours, _ = cv2.findContours(ref_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    test_contours, _ = cv2.findContours(test_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter small contours (noise)
    ref_contours = [c for c in ref_contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]
    test_contours = [c for c in test_contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]

    # Compute absolute difference to find defect regions
    diff = cv2.absdiff(ref_edges, test_edges)
    diff_dilated = cv2.dilate(diff, None, iterations=3)
    diff_contours, _ = cv2.findContours(diff_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    significant_diffs = [c for c in diff_contours if cv2.contourArea(c) > MIN_CONTOUR_AREA]

    return {
        'ref_contour_count': len(ref_contours),
        'test_contour_count': len(test_contours),
        'diff_count': len(significant_diffs),
        'diff_contours': significant_diffs,
        'diff_image': diff
    }


def create_diff_visualization(ref_path, test_path, ssim_diff_map, contour_data):
    """Create a side-by-side diff visualization with highlighted defects."""
    ref = cv2.imread(ref_path)
    test = cv2.imread(test_path)
    test = cv2.resize(test, (ref.shape[1], ref.shape[0]))

    # Create a copy to draw on
    overlay = test.copy()

    # Highlight contour differences in red
    for contour in contour_data['diff_contours']:
        cv2.drawContours(overlay, [contour], -1, (0, 0, 255), 3)
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(overlay, (x - 5, y - 5), (x + w + 5, y + h + 5), (0, 0, 255), 2)
        cv2.putText(overlay, "DEFECT", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Create SSIM heatmap
    ssim_heatmap = cv2.applyColorMap(255 - ssim_diff_map, cv2.COLORMAP_JET)
    ssim_heatmap = cv2.resize(ssim_heatmap, (ref.shape[1], ref.shape[0]))

    # Stack: Reference | Test | Overlay | SSIM Heatmap
    h, w = ref.shape[:2]
    label_h = 40
    canvas = np.zeros((h + label_h, w * 4, 3), dtype=np.uint8)

    # Add images
    canvas[label_h:, :w] = ref
    canvas[label_h:, w:2*w] = test
    canvas[label_h:, 2*w:3*w] = overlay
    canvas[label_h:, 3*w:4*w] = ssim_heatmap

    # Add labels
    labels = ["REFERENCE", "TEST", "DEFECTS", "SSIM HEATMAP"]
    colors = [(0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 0, 255)]
    for i, (label, color) in enumerate(zip(labels, colors)):
        cv2.putText(canvas, label, (i * w + 10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return canvas


# ═══════════════════════════════════════════
#  MAIN ANALYSIS
# ═══════════════════════════════════════════

def print_header(ref_path, test_path):
    print()
    print("=" * 60)
    print("  🏭 VISION MACHINE DETECTOR - Defect Analysis")
    print("  🧠 Powered by ResNet50 CNN + OpenCV")
    print("=" * 60)
    print()
    print(f"  📸 Reference:  {os.path.basename(ref_path)}")
    print(f"  🔍 Test:       {os.path.basename(test_path)}")
    print()


def print_method(number, name, details, passed):
    status = "✅ OK" if passed else "⚠️  DEFECT DETECTED"
    print(f"  ── Method {number}: {name} ──")
    for key, value in details.items():
        print(f"     {key}: {value}")
    print(f"     Status: {status}")
    print()


def analyze(ref_path, test_path, save_diff=False, output_dir="output"):
    """Run full defect analysis on reference vs test image."""

    # Validate inputs
    if not os.path.exists(ref_path):
        print(f"  ❌ Reference image not found: {ref_path}")
        sys.exit(1)
    if not os.path.exists(test_path):
        print(f"  ❌ Test image not found: {test_path}")
        sys.exit(1)

    print_header(ref_path, test_path)

    results = {}

    # ── Method 1: CNN Deep Features ──
    print("  🔄 Running CNN analysis...")
    extractor = CNNFeatureExtractor()
    print()

    ref_features = extractor.extract_features(ref_path)
    test_features = extractor.extract_features(test_path)
    cnn_similarity = extractor.cosine_similarity(ref_features, test_features)
    cnn_passed = cnn_similarity >= CNN_SIMILARITY_THRESHOLD

    print_method(1, "CNN Deep Features (ResNet50)", {
        "Cosine Similarity": f"{cnn_similarity:.4f}",
        "Threshold": f"{CNN_SIMILARITY_THRESHOLD}",
        "Interpretation": "High-level pattern & structure comparison"
    }, cnn_passed)
    results['cnn'] = cnn_passed

    # ── Method 2: SSIM ──
    ssim_score, ssim_diff = compute_ssim(ref_path, test_path)
    ssim_passed = ssim_score >= SSIM_THRESHOLD

    print_method(2, "SSIM (Structural Similarity)", {
        "Score": f"{ssim_score:.4f} ({ssim_score*100:.1f}%)",
        "Threshold": f"{SSIM_THRESHOLD}",
        "Interpretation": "Pixel-structure comparison"
    }, ssim_passed)
    results['ssim'] = ssim_passed

    # ── Method 3: Perceptual Difference (CNN Multi-Layer) ──
    ref_layers = extractor.extract_layer_features(ref_path)
    test_layers = extractor.extract_layer_features(test_path)
    perceptual_diff = extractor.perceptual_difference(ref_layers, test_layers)
    perceptual_passed = perceptual_diff <= PERCEPTUAL_DIFF_THRESHOLD

    print_method(3, "Perceptual Difference (CNN Multi-Layer)", {
        "Difference Score": f"{perceptual_diff:.6f}",
        "Threshold": f"{PERCEPTUAL_DIFF_THRESHOLD}",
        "Interpretation": "Multi-scale pattern difference (edges → shapes → structure)"
    }, perceptual_passed)
    results['perceptual'] = perceptual_passed

    # ── Method 4: Contour Analysis ──
    contour_data = analyze_contours(ref_path, test_path)
    contour_passed = contour_data['diff_count'] <= CONTOUR_DIFF_THRESHOLD

    print_method(4, "Contour Analysis (OpenCV)", {
        "Reference contours": str(contour_data['ref_contour_count']),
        "Test contours": str(contour_data['test_contour_count']),
        "Significant differences": str(contour_data['diff_count']),
        "Threshold": f"{CONTOUR_DIFF_THRESHOLD}",
        "Interpretation": "Edge/shape break detection"
    }, contour_passed)
    results['contour'] = contour_passed

    # ── Final Verdict ──
    failed_count = sum(1 for v in results.values() if not v)
    total = len(results)
    all_passed = failed_count == 0

    print("  " + "=" * 56)
    if all_passed:
        print("  ══ FINAL VERDICT ══")
        print(f"  ✅ ALL CLEAR - Pattern matches reference ({total}/{total} methods passed)")
    else:
        print("  ══ FINAL VERDICT ══")
        print(f"  ⚠️  DEFECT DETECTED ({failed_count}/{total} methods flagged issues)")
        print()
        print("  Failed methods:")
        method_names = {
            'cnn': 'CNN Deep Features',
            'ssim': 'SSIM',
            'perceptual': 'Perceptual Difference',
            'contour': 'Contour Analysis'
        }
        for key, passed in results.items():
            if not passed:
                print(f"    ❌ {method_names[key]}")

    print("  " + "=" * 56)

    # ── Save visualization ──
    if save_diff:
        os.makedirs(output_dir, exist_ok=True)
        diff_image = create_diff_visualization(ref_path, test_path, ssim_diff, contour_data)
        output_path = os.path.join(output_dir, "diff_visualization.jpg")
        cv2.imwrite(output_path, diff_image)
        print(f"\n  💾 Saved diff visualization → {output_path}")

    print()
    return results


# ═══════════════════════════════════════════
#  CLI ENTRY POINT
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🏭 Vision Machine Detector - Factory Defect Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python detector.py --reference ref.jpg --test test.jpg
  python detector.py --reference ref.jpg --test test.jpg --save-diff
  python detector.py -r images/good.jpg -t images/current.jpg --save-diff --output results/
        """
    )
    parser.add_argument('-r', '--reference', required=True,
                        help='Path to the reference (good) image')
    parser.add_argument('-t', '--test', required=True,
                        help='Path to the test (current) image to check')
    parser.add_argument('--save-diff', action='store_true',
                        help='Save a diff visualization image')
    parser.add_argument('--output', default='output',
                        help='Output directory for diff images (default: output/)')

    args = parser.parse_args()
    analyze(args.reference, args.test, args.save_diff, args.output)


if __name__ == "__main__":
    main()
