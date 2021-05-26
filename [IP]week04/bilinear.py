import cv2
import numpy as np


def my_bilinear(src, scale):
    (h, w) = src.shape
    h_dst = int(h * scale + 0.5)
    w_dst = int(w * scale + 0.5)

    dst = np.zeros((h_dst, w_dst))

    for row in range(h_dst):
        for col in range(w_dst):
            row_before = row/scale
            col_before = col/scale

            if row_before % 1 == 0 and col_before % 1 == 0:
                dst[row][col] = src[int(row_before)][int(col_before)]
            else:
                m = int(col_before)
                n = int(row_before)
                s = col_before - m
                t = row_before - n
                if m<w-1 and n<h-1:
                    dst_img = (1-s)*(1-t)*src[n][m] + s*(1-t)*src[n][m+1] + (1-s)*t*src[n+1][m] + s*t*src[n+1][m+1]
                dst[row][col] = dst_img
    return dst


if __name__ == '__main__':
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)

    scale = 1/2

    my_dst_mini = my_bilinear(src, scale)
    my_dst_mini = my_dst_mini.astype(np.uint8)

    my_dst = my_bilinear(my_dst_mini, 1/scale)
    my_dst = my_dst.astype(np.uint8)

    cv2.imshow('original', src)
    cv2.imshow('my bilinear mini', my_dst_mini)
    cv2.imshow('my bilinear', my_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()
