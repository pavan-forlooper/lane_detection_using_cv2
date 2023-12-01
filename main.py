import numpy as np
import matplotlib.image as mpimg
from helping_functions import Help


def main():
    image = mpimg.imread('original_lane_detection_5.jpg')
    helping_functions_instance = Help()
    # helping_functions_instance.print_image(image)

    gray_img = helping_functions_instance.gray_scale(image)
    # helping_functions_instance.print_image(gray_img)

    blur_img = helping_functions_instance.apply_blur(gray_img)
    # helping_functions_instance.print_image(blur_img)

    edge_img = helping_functions_instance.edge_detection(blur_img, 30, 200)
    # helping_functions_instance.print_image(edge_img)

    vertices = np.array([[(0, image.shape[0]), (300, 176), (350, 176), (image.shape[1], image.shape[0])]],
                        dtype=np.int32)
    roi_img = helping_functions_instance.region_of_interest(edge_img, vertices)
    # helping_functions_instance.print_image(roi_img)
    # helping_functions_instance.print_image(helping_functions_instance.region_of_interest(image, vertices))

    rho = 1
    theta = np.pi / 720
    threshold = 20
    min_line_len = 0
    max_line_gap = 10
    hough_img = helping_functions_instance.hough_lines(roi_img, rho, theta, threshold, min_line_len, max_line_gap)
    # helping_functions_instance.print_image(hough_img)

    final_img = helping_functions_instance.weighted_img(hough_img, image)
    helping_functions_instance.print_image(final_img)



if __name__ == '__main__':
    main()
