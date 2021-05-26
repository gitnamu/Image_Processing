import cv2
import numpy as np


def dilation(B, S):
    h,w = B.shapea
    S_h, S_w = S.shape
    S_h_half = S_h//2
    S_w_half = S_w // 2
    dst = B.copy()

    for row in range(h):
        for col in range(w):
            if B[row][col] != 1:
                continue

            row_min = row - S_h_half
            row_max = row + S_h_half
            col_min = col - S_w_half
            col_max = col + S_w_half
            S_row_min = 0
            S_row_max = S_h - 1
            S_col_min = 0
            S_col_max = S_w - 1

            if row_min < 0:
                row_min = 0
                S_row_min = -(row - S_h_half)
            if row_max > h-1:
                row_max = h-1
                S_row_max = row + S_h_half - (h - 1)
            if col_min < 0 :
                col_min = 0
                S_col_min = -(col - S_w_half)
            if col_max > w-1:
                col_max = w-1
                S_col_max = col + S_w_half - (w - 1)

            dst[row_min:row_max+1, col_min:col_max+1] += S[S_row_min:S_row_max+1, S_col_min:S_col_max+1]
    dst[dst > 0] = 1
    return dst


def erosion(B, S):
    h,w = B.shape
    S_h, S_w = S.shape
    S_h_half = S_h // 2
    S_w_half = S_w // 2
    dst = B.copy()

    for row in range(h):
        for col in range(w):
            if B[row][col] != 1:
                continue

            if row - S_h_half < 0 or row + S_h_half > h-1 or col - S_w_half < 0 or col + S_w_half > w-1:
                dst[row][col] = 0
                continue

            tmp = B[row - S_h_half:row + S_h_half + 1, col - S_w_half:col + S_w_half + 1] - S
            if np.sum(tmp) < 0:
                dst[row][col] = 0
    return dst


def opening(B, S):
    erosion_img = erosion(B, S)
    dst = dilation(erosion_img, S)

    return dst


def closing(B, S):
    dilation_img = dilation(B, S)
    dst = erosion(dilation_img, S)

    return dst


if __name__ == '__main__':
    B = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]])

    S = np.array(
        [[1, 1, 1],
         [1, 1, 1],
         [1, 1, 1]])

    cv2.imwrite('morphology_B.png', (B*255).astype(np.uint8))

    img_dilation = dilation(B, S)
    img_dilation = (img_dilation*255).astype(np.uint8)
    print(img_dilation)
    cv2.imwrite('morphology_dilation.png', img_dilation)

    img_erosion = erosion(B, S)
    img_erosion = (img_erosion * 255).astype(np.uint8)
    print(img_erosion)
    cv2.imwrite('morphology_erosion.png', img_erosion)

    img_opening = opening(B, S)
    img_opening = (img_opening * 255).astype(np.uint8)
    print(img_opening)
    cv2.imwrite('morphology_opening.png', img_opening)

    img_closing = closing(B, S)
    img_closing = (img_closing * 255).astype(np.uint8)
    print(img_closing)
    cv2.imwrite('morphology_closing.png', img_closing)
