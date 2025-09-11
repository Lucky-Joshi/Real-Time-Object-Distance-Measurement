# Real-Time Object Distance Measurement

This Python script utilizes **OpenCV** to perform real-time detection of yellow objects from a webcam feed and calculate the physical distance between them. It works by identifying the center points of the objects and converting the pixel distance to a real-world distance using a calibrated factor.

-----

## Features

  - **Real-time object detection:** Continuously processes video from a webcam.
  - **Color-based filtering:** Uses HSV color space to isolate yellow objects.
  - **Distance calculation:** Measures the distance in centimeters between adjacent detected objects.
  - **Live visualization:** Displays the original feed with bounding boxes, labels, and distance measurements.

-----

## Prerequisites

Before you begin, ensure you have the following installed:

  - **Python 3.x**
  - **OpenCV:** `pip install opencv-python`
  - **NumPy:** `pip install numpy`

-----

## Getting Started

### 1\. Calibration

The `DISTANCE_THRESHOLD` variable is crucial for accurate measurements. You must calibrate it for your specific camera setup. To do this, place two yellow objects at a known distance (e.g., 10 cm) in front of the camera, run the script, and adjust `DISTANCE_THRESHOLD` until the displayed distance is correct.

### 2\. Running the Script

1.  Save the code as a Python file (e.g., `distance_meter.py`).
2.  Open your terminal or command prompt.
3.  Run the script using the following command:
    ```bash
    python distance_meter.py
    ```
4.  The script will automatically try to find and connect to your webcam.
5.  Press `q` on the keyboard to exit the application.

-----

## How It Works

The script follows these key steps:

1.  **Camera Initialization:** It tries to open a webcam, iterating through a few camera indices to find a valid one.
2.  **Color Filtering:** It converts each video frame from BGR to the **HSV** color space, which is better for color-based detection. It then creates a binary mask to isolate pixels that fall within the defined yellow hue range.
3.  **Contour Detection:** The script finds **contours** (outlines of shapes) in the binary mask. It filters out small contours to ignore noise.
4.  **Object Identification:** For each significant contour, it draws a bounding box, calculates the center point, and labels the object.
5.  **Distance Calculation:** It measures the Euclidean distance in pixels between the center points of adjacent detected objects.
6.  **Conversion & Display:** The pixel distance is converted to a real-world distance (in cm) using the `DISTANCE_THRESHOLD` and displayed on the screen next to a connecting line between the objects.