"""
Generate synthetic test images to demonstrate the defect detector.
Creates a reference image and two test images:
  - test_good.jpg: similar to reference (should pass)
  - test_defect.jpg: with visible damage (should fail)
"""

import cv2
import numpy as np
import os


def create_cork_pattern(width=640, height=480, seed=42):
    """Create a synthetic cork-like pattern with a wavy shape (مصورة)."""
    np.random.seed(seed)
    img = np.ones((height, width, 3), dtype=np.uint8) * 200  # light gray background

    # Draw cork-like texture (brownish with noise)
    cork_color = np.array([140, 180, 210], dtype=np.uint8)  # BGR brownish
    noise = np.random.randint(-20, 20, (height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw the main pattern shape (المصورة) - a wavy ridge pattern
    for i in range(3):
        y_center = 120 + i * 120
        points = []
        for x in range(0, width, 5):
            y = int(y_center + 15 * np.sin(x * 0.02 + i * 0.5))
            points.append([x, y])

        pts = np.array(points, dtype=np.int32)

        # Draw thick wavy line (the pattern/shape)
        cv2.polylines(img, [pts], False, (80, 120, 160), thickness=12)
        # Add inner detail
        cv2.polylines(img, [pts + [0, 3]], False, (100, 140, 180), thickness=6)

    # Add some grid pattern (cork structure)
    for x in range(0, width, 40):
        cv2.line(img, (x, 0), (x, height), (180, 190, 195), 1)
    for y in range(0, height, 40):
        cv2.line(img, (0, y), (width, y), (180, 190, 195), 1)

    # Add machine frame borders
    cv2.rectangle(img, (20, 20), (width - 20, height - 20), (60, 60, 60), 3)

    return img


def create_good_test(reference):
    """Create a test image that's very similar to reference (minor natural variation)."""
    img = reference.copy()

    # Add slight noise (natural camera variation)
    noise = np.random.randint(-5, 5, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Slight brightness variation
    img = np.clip(img.astype(np.int16) + 3, 0, 255).astype(np.uint8)

    return img


def create_defect_test(reference):
    """Create a test image with visible defects (breaks in the pattern)."""
    img = reference.copy()

    # Defect 1: Break in the pattern (المصورة انقطعت)
    # White out a section of the first wave
    cv2.rectangle(img, (180, 100), (280, 155), (200, 200, 200), -1)
    # Add some noise to make it look natural
    noise_region = np.random.randint(-10, 10, (55, 100, 3), dtype=np.int16)
    img[100:155, 180:280] = np.clip(img[100:155, 180:280].astype(np.int16) + noise_region, 0, 255).astype(np.uint8)

    # Defect 2: Deformation in the second wave (المادة راخية)
    for x in range(350, 480):
        for y_offset in range(-8, 20):
            y = int(240 + 15 * np.sin(x * 0.02 + 0.5)) + y_offset
            if 0 <= y < img.shape[0]:
                img[y, x] = [190, 195, 200]  # fade out the pattern

    # Defect 3: Missing section in third wave
    cv2.rectangle(img, (400, 335), (520, 395), (195, 200, 205), -1)

    # Defect 4: Dark spot (material buildup or contamination)
    cv2.circle(img, (100, 300), 25, (40, 50, 60), -1)

    return img


def main():
    output_dir = os.path.join(os.path.dirname(__file__), "images")
    os.makedirs(output_dir, exist_ok=True)

    print("🏭 Generating synthetic factory test images...")
    print()

    # Create reference (good) pattern
    reference = create_cork_pattern()
    ref_path = os.path.join(output_dir, "reference.jpg")
    cv2.imwrite(ref_path, reference)
    print(f"  ✅ Reference image saved: {ref_path}")

    # Create good test (should pass)
    good_test = create_good_test(reference)
    good_path = os.path.join(output_dir, "test_good.jpg")
    cv2.imwrite(good_path, good_test)
    print(f"  ✅ Good test image saved: {good_path}")

    # Create defective test (should fail)
    defect_test = create_defect_test(reference)
    defect_path = os.path.join(output_dir, "test_defect.jpg")
    cv2.imwrite(defect_path, defect_test)
    print(f"  ✅ Defect test image saved: {defect_path}")

    print()
    print("📋 Next steps:")
    print(f"  1. Test with good image:")
    print(f"     python detector.py -r {ref_path} -t {good_path} --save-diff")
    print(f"  2. Test with defect image:")
    print(f"     python detector.py -r {ref_path} -t {defect_path} --save-diff")
    print()
    print("  Or use your own factory images by replacing the files in images/")


if __name__ == "__main__":
    main()
