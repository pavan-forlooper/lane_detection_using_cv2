# lane_detection_using_cv2
This Python program detects lanes in an image using computer vision techniques. It takes an input image, performs various image processing operations, and highlights the detected lanes.


# Lane Detection Program

This Python program detects lanes in an image using computer vision techniques. It takes an input image, performs various image processing operations, and highlights the detected lanes.

## Features
- Gray scaling
- Gaussian blur
- Canny edge detection
- Region of interest selection
- Hough line detection
- Drawing lane lines on the original image

## Dependencies
- NumPy
- Matplotlib
- OpenCV

## Installation
1. Clone the repository:

    ```bash
    git clone https://github.com/your-username/lane-detection.git
    cd lane-detection
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Run the main program:

    ```bash
    python main.py
    ```

2. The final image with detected lanes will be saved in the current directory as `final_image.jpg`.

## Configuration
You can tweak parameters in the `main.py` script to adjust the detection algorithm according to your needs.

## Contributing
Contributions are welcome! Feel free to open issues, submit pull requests, or suggest improvements.
