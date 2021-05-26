import cv2
import numpy as np


def C(w, n = 8):
    if w == 0:
        return (1/n)**0.5
    else:
        return (2/n)**0.5


def Spatial2Frequency_mask(block, n = 8):
    dst = np.zeros(block.shape)
    v, u = dst.shape

    y, x = np.mgrid[0:u, 0:v]

    mask = np.zeros((n*n, n*n))

    for v_ in range(v):
        for u_ in range(u):
            tmp = block * np.cos(((2*x+1)*u_*np.pi)/(2*n)) * np.cos(((2*y+1)*v_*np.pi)/(2*n))
            tmp = my_normalize(tmp)
            mask[v_*4:(v_+1)*4, u_*4:(u_+1)*4] = tmp

    return mask


def my_normalize(src):
    dst = src.copy()

    if np.max(dst) > np.min(dst):
        dst = dst - np.min(dst)

    dst = (dst / np.max(dst) * 255).astype(np.uint8)

    return dst


if __name__ == '__main__':
    block_size = 4
    src = np.ones((block_size, block_size))

    mask = Spatial2Frequency_mask(src, n=block_size)
    mask = mask.astype(np.uint8)
    print(mask)

    mask = cv2.resize(mask, (320, 320), interpolation=cv2.INTER_NEAREST)

    cv2.imshow('201701982 mask', mask)
    cv2.waitKey()
    cv2.destroyAllWindows()
