import numpy as np
import cv2
import matplotlib.pyplot as plt

def my_calcHist(src):
    h,w = src.shape[:2]
    hist = np.zeros((256,), dtype=int)
    for row in range(h):
        for col in range(w):
            intensity = src[row,col]
            hist[intensity] += 1
    return hist

def my_normalize_hist(hist, pixel_num):
    normalized_hist = hist / pixel_num

    return normalized_hist


def my_PDF2CDF(pdf):
    pdf = np.array(pdf)
    cdf = np.cumsum(pdf)

    return cdf


def my_denormalize(normalized, gray_level):
    denormalized = np.zeros((len(normalized),), dtype=int)
    for i in range(len(normalized)):
        denormalized[i] = normalized[i] * gray_level

    return denormalized


def my_calcHist_equalization(denormalized, hist):
    hist_equal = np.zeros((len(hist),), dtype=int)
    for i in range(len(denormalized)):
        hist_equal[denormalized[i]] += hist[i]
    return hist_equal


def my_equal_img(src, output_gray_level):
    h,w = src.shape[:2]
    dst = np.zeros((h,w), dtype=np.uint8)
    for row in range(h):
        for col in range(w):
            dst[row,col] = output_gray_level[src[row,col]]
    return dst


def my_hist_equal(src):
    (h, w) = src.shape
    max_gray_level = 255
    histogram = my_calcHist(src)
    normalized_histogram = my_normalize_hist(histogram, h * w)
    normalized_output = my_PDF2CDF(normalized_histogram)
    denormalized_output = my_denormalize(normalized_output, max_gray_level)
    output_gray_level = denormalized_output.astype(int)
    hist_equal = my_calcHist_equalization(output_gray_level, histogram)

    plt.plot(output_gray_level, )
    plt.title('mapping function')
    plt.xlabel('input intensity')
    plt.ylabel('output intensity')
    plt.show()

    dst = my_equal_img(src, output_gray_level)

    return dst, hist_equal

if __name__ == '__main__':
    src = cv2.imread('fruits_div3.jpg', cv2.IMREAD_GRAYSCALE)
    hist = my_calcHist(src)
    dst, hist_equal = my_hist_equal(src)

    plt.figure(figsize=(8, 5))
    cv2.imshow('original', src)
    binX = np.arange(len(hist))
    plt.title('my histogram')
    plt.bar(binX, hist, width=0.5, color='g')
    plt.show()

    plt.figure(figsize=(8, 5))
    cv2.imshow('equalizetion after image', dst)
    binX = np.arange(len(hist_equal))
    plt.title('my histogram equalization')
    plt.bar(binX, hist_equal, width=0.5, color='g')
    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()
