import cv2
import numpy as np
import math as m

# --- Configuration ---
DISTANCE_THRESHOLD = 0.06912  # Calibrated distance factor (REQUIRED)
MIN_CONTOUR_AREA = 500  # Minimum contour area to filter noise
YELLOW_HUE_LOWER = 20  # Lower bound for yellow hue
YELLOW_HUE_UPPER = 30  # Upper bound for yellow hue
FONT_SCALE = 0.7  # Font size for distance display
FONT_THICKNESS = 2  # Font thickness for distance display
CAMERA_INDEX = 0  # Initial camera index to try

def open_camera(index):
    """
    Opens the camera at the given index and returns the capture object.
    Returns None if the camera cannot be opened.
    """
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Could not open camera at index {index}.")
        return None
    return cap

def process_frame(img, distance_threshold):
    """
    Processes a single frame to detect yellow objects, calculate distances, and display results.

    Args:
        img: The input frame (BGR image).
        distance_threshold: Calibration factor for distance calculation.

    Returns:
        The processed image with detections and distance measurements.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_yellow = np.array([YELLOW_HUE_LOWER, 100, 100])
    upper_yellow = np.array([YELLOW_HUE_UPPER, 255, 255])

    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    result = cv2.bitwise_and(img, img, mask=mask)  # Useful for debugging

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    XY = []  # List to store the center points of detected objects

    le = 65  # ASCII for 'A', used for labeling objects

    for i in contours:
        if cv2.contourArea(i) > MIN_CONTOUR_AREA:
            x, y, w, h = cv2.boundingRect(i)
            cx = x + w // 2
            cy = y + h // 2
            XY.append([cx, cy])

            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)
            cv2.putText(img, chr(le), (cx - 10, cy - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            le += 1  # Increment label

    # Calculate and display distances between adjacent objects
    for i in range(len(XY) - 1):
        x1, y1 = XY[i]
        x2, y2 = XY[i + 1]
        distance_px = m.sqrt((x2 - x1)**2 + (y2 - y1)**2) #Distance in pixel
        distance_cm = distance_px * distance_threshold  # Convert to cm

        tx = (x1 + x2) // 2
        ty = (y1 + y2) // 2
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f'{distance_cm:.2f} cm', (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 255), FONT_THICKNESS)

    return img, hsv, mask, result

def main():
    """
    Main function to capture video, process frames, and display results.
    """
    camera_index = 0 #Start with index 0

    #Try each camera index until success.
    frame = None
    while True:
        frame = open_camera(camera_index)
        if frame is not None:
            break #Successful camera found.
        camera_index +=1
        if camera_index >= 10: #Limit number of attempts.
            print("Tried too many camera indices. Exiting")
            return #Give up after 10 attempts.

    if not frame.isOpened():
        print(f"Could not open camera at any index. Exiting.")
        return

    while True:
        ok, img = frame.read()

        if not ok:
            print("Error reading frame.  Check camera connection and/or index.")
            break

        processed_img, hsv, mask, result = process_frame(img, DISTANCE_THRESHOLD)

        cv2.imshow("Processed Output", processed_img)
        cv2.imshow("HSV", hsv)
        cv2.imshow("Mask", mask)
        cv2.imshow("Result", result)

        if cv2.waitKey(1) == ord("q"):
            break

    frame.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
