"""Quick script to find available cameras."""
import cv2

print("Scanning for cameras...")
for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"  Camera {i}: FOUND ({w}x{h})")
        cap.release()
    else:
        print(f"  Camera {i}: not found")

print("\nUse: python camera.py -r images/reference_straight.png --camera <ID>")
