from collections import deque
import cv2
import numpy as np
from my_library.filtering import my_filtering


def apply_lowNhigh_pass_filter(src, fsize, sigma=1):
    y, x = np.mgrid[-(fsize // 2):(fsize // 2) + 1, -(fsize // 2):(fsize // 2) + 1]

    DoG_x = -(x / (sigma ** 2)) * np.exp(-((x ** 2) + (y ** 2)) / (2 * (sigma ** 2)))
    DoG_y = -(y / (sigma ** 2)) * np.exp(-((x ** 2) + (y ** 2)) / (2 * (sigma ** 2)))

    Ix = my_filtering(src, DoG_x, 'zero')
    Iy = my_filtering(src, DoG_y, 'zero')

    return Ix, Iy


def calcMagnitude(Ix, Iy):
    magnitude = np.sqrt(Ix**2 + Iy**2)

    return magnitude


def calcAngle(Ix, Iy):
    e = 1E-6
    angle = np.arctan(Iy/(Ix+e))

    return angle


def non_maximum_supression(magnitude, angle):
    h, w = magnitude.shape
    largest_magnitude = np.zeros((h+1, w+1))

    for row in range(1, h-1):
        for col in range(1, w-1):
            gradient_1 = 0
            gradient_2 = 0
            if (-np.pi/2) <= angle[row][col] < (-np.pi/4):
                distance = np.tan(np.pi/2 + angle[row][col])
                gradient_1 = distance * magnitude[row - 1][col + 1] + (1 - distance) * magnitude[row - 1][col]
                gradient_2 = distance * magnitude[row + 1][col - 1] + (1 - distance) * magnitude[row + 1][col]
            elif (-np.pi/4) <= angle[row][col] < 0:
                distance = np.tan(-angle[row][col])
                gradient_1 = distance * magnitude[row - 1][col + 1] + (1 - distance) * magnitude[row][col + 1]
                gradient_2 = distance * magnitude[row + 1][col - 1] + (1 - distance) * magnitude[row][col - 1]
            elif 0 <= angle[row][col] < (np.pi/4):
                distance = np.tan(angle[row][col])
                gradient_1 = distance * magnitude[row + 1][col + 1] + (1 - distance) * magnitude[row][col + 1]
                gradient_2 = distance * magnitude[row - 1][col - 1] + (1 - distance) * magnitude[row][col - 1]
            elif (np.pi/4) <= angle[row][col] < (np.pi/2):
                distance = np.tan(np.pi/2 - angle[row][col])
                gradient_1 = distance * magnitude[row + 1][col + 1] + (1 - distance) * magnitude[row + 1][col]
                gradient_2 = distance * magnitude[row - 1][col - 1] + (1 - distance) * magnitude[row - 1][col]

            if magnitude[row][col] > gradient_1 and magnitude[row][col] > gradient_2:
                largest_magnitude[row][col] = magnitude[row][col]
            else:
                largest_magnitude[row][col] = 0

    return largest_magnitude


def double_thresholding(src):
    dst = src.copy()

    dst -= dst.min()
    dst /= dst.max()
    dst *= 255
    dst = dst.astype(np.uint8)

    (h, w) = dst.shape

    high_threshold_value, _ = cv2.threshold(dst, 0, 255, cv2.THRESH_OTSU)
    low_threshold_value = high_threshold_value * 0.4

    white_row, white_col = np.where(src > high_threshold_value)
    black_row, black_col = np.where(src < low_threshold_value)
    q_row = deque(white_row)
    q_col = deque(white_col)

    dst[white_row, white_col] = 255
    dst[black_row, black_col] = 0

    around_pixels = [[0, -1], [0, 1], [-1, 0], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1]]

    while q_row and q_col:
        row = q_row.popleft()
        col = q_col.popleft()

        for y, x in around_pixels:
            if (row+y) < 0 or h <= (row+y) or (col+x) < 0 or w <= (col+x):
                continue
            if low_threshold_value <= dst[row+y][col+x] <= high_threshold_value:
                dst[row + y][col + x] = 255
                q_row.appendleft(row + y)
                q_col.appendleft(col + x)
    dst[dst < 255] = 0

    return dst


def my_canny_edge_detection(src, fsize=3, sigma=1):
    Ix, Iy = apply_lowNhigh_pass_filter(src, fsize, sigma)

    Ix_t = np.abs(Ix)
    Iy_t = np.abs(Iy)
    Ix_t = Ix_t / Ix_t.max()
    Iy_t = Iy_t / Iy_t.max()

    cv2.imshow("Ix", Ix_t)
    cv2.imshow("Iy", Iy_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    magnitude = calcMagnitude(Ix, Iy)
    angle = calcAngle(Ix, Iy)

    magnitude_t = magnitude
    magnitude_t = magnitude_t / magnitude_t.max()
    cv2.imshow("magnitude", magnitude_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    largest_magnitude = non_maximum_supression(magnitude, angle)

    largest_magnitude_t = largest_magnitude
    largest_magnitude_t = largest_magnitude_t / largest_magnitude_t.max()
    cv2.imshow("largest_magnitude", largest_magnitude_t)
    cv2.waitKey()
    cv2.destroyAllWindows()

    dst = double_thresholding(largest_magnitude)

    return dst


def main():
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_canny_edge_detection(src)
    cv2.imshow('original', src)
    cv2.imshow('my canny edge detection', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()