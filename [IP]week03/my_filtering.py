import cv2
import numpy as np


def my_padding(src, pad_shape, pad_type='zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    pad_img[p_h:p_h + h, p_w:p_w + w] = src

    if pad_type == 'repetition':
        print('repetition padding')
        # up
        pad_img[:p_h, :] = pad_img[p_h, :]
        # down
        pad_img[p_h + h:, :] = pad_img[p_h + h-1, :]
        # left
        pad_img[:,:p_w] = pad_img[:,p_w:p_w + 1]
        #right
        pad_img[:,p_w + w:] = pad_img[:,p_w + w - 1:p_w + w]

    else:
        print('zero padding')

    return pad_img


def my_filtering(src, ftype, fshape, pad_type='zero'):
    (h, w) = src.shape
    src_pad = my_padding(src, (fshape[0] // 2, fshape[1] // 2), pad_type)
    dst = np.zeros((h, w))

    if ftype == 'average':
        print('average filtering')

        mask = np.full(fshape, 1 / (fshape[0] * fshape[1]))

        print(mask)

    elif ftype == 'sharpening':
        print('sharpening filtering')

        tmp_mask_1 = np.full(fshape, 1 / (fshape[0] * fshape[1]))
        tmp_mask_2 = np.zeros(fshape)
        tmp_mask_2[fshape[0] // 2, fshape[1] // 2] = 2

        mask = tmp_mask_2 - tmp_mask_1

        print(mask)

    msk_h = fshape[0] // 2
    msk_w = fshape[1] // 2
    for i in range(msk_h, h):
        for j in range(msk_w, w):
            filtered = min(255, np.sum(src_pad[i - msk_h:i + msk_h + 1, j - msk_w:j + msk_w + 1] * mask))
            filtered = max(0, filtered)
            dst[i, j] = filtered
    dst = (dst + 0.5).astype(np.uint8)
    return dst


if __name__ == '__main__':
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)

    rep_test = my_padding(src, (20, 20), 'repetition')
    dst_average = my_filtering(src, 'average', (11, 13), 'repetition')
    dst_sharpening = my_filtering(src, 'sharpening', (11, 13), 'repetition')

    cv2.imshow('original', src)
    cv2.imshow('average filter', dst_average)
    cv2.imshow('sharpening filter', dst_sharpening)
    cv2.imshow('repetition padding test', rep_test.astype(np.uint8))
    cv2.waitKey()
    cv2.destroyAllWindows()
