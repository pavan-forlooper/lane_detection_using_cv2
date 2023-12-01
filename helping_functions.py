import cv2
import matplotlib.pyplot as plt
import numpy as np


class Help:

    def print_image(self, image):
        print(image.shape)
        plt.imshow(image, cmap='gray')
        plt.show()

    def gray_scale(self, image):
        return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    def apply_blur(self, image):
        return cv2.GaussianBlur(image, (7, 7), 0)

    def edge_detection(self, image, lower_threshold, upper_threshold):
        return cv2.Canny(image, lower_threshold, upper_threshold)

    def region_of_interest(self, img, vertices):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, vertices, 255)
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

    def draw_lines(self, img, lines, color=[255, 0, 0], thickness=6):

        # list to get positives and negatives values

        x_bottom_pos = []
        x_upperr_pos = []
        x_bottom_neg = []
        x_upperr_neg = []

        y_bottom = 300
        y_upperr = 175

        # y1 = slope*x1 + b
        # b = y1 - slope*x1
        # y = slope*x + b
        # x = (y - b)/slope

        slope = 0
        b = 0

        # get x upper and bottom to lines with slope positive and negative
        for line in lines:
            for x1, y1, x2, y2 in line:
                # cv2.imshow('image', img)
                # cv2.waitKey(0)
                # test and filter values to slope

                slope = ((y2 - y1) / (x2 - x1))

                if 0.4 < slope < 0.9:
                    b = y1 - slope * x1

                    x_bottom_pos.append((y_bottom - b) / slope)
                    x_upperr_pos.append((y_upperr - b) / slope)

                elif -0.4 > slope > -0.9:

                    b = y1 - slope * x1

                    x_bottom_neg.append((y_bottom - b) / slope)
                    x_upperr_neg.append((y_upperr - b) / slope)
                #cv2.imshow('image', img)
                #cv2.waitKey(0)
                #cv2.line(img, (x1, y1), (x2, y2), color, thickness)
                # print(x1, y1, x2, y2)
            print(x_bottom_neg, x_upperr_neg)

        # creating a new 2d array with means
        lines_mean = np.array(
            [[int(np.mean(x_bottom_pos)), int(np.mean(y_bottom)), int(np.mean(x_upperr_pos)), int(np.mean(y_upperr))],
             [int(np.mean(x_bottom_neg)), int(np.mean(y_bottom)), int(np.mean(x_upperr_neg)), int(np.mean(y_upperr))]])

        # Drawing the lines
        for i in range(len(lines_mean)):
            # cv2.imshow('image', img)
            # cv2.waitKey(0)
            cv2.line(img, (lines_mean[i, 0], lines_mean[i, 1]), (lines_mean[i, 2], lines_mean[i, 3]), color, thickness)

    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                                maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        self.draw_lines(line_img, lines)
        # print(lines)
        # cv2.imshow(lines)
        # cv2.waitKey(0)
        return line_img

    def weighted_img(self, img, initial_img, α=0.8, β=1, λ=0):
        cv2.imwrite('output.jpg', cv2.addWeighted(initial_img, α, img, β, λ))
        return cv2.addWeighted(initial_img, α, img, β, λ)
