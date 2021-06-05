import numpy as np
import cv2

def forward(src, M, fit=False):
    print('< forward >')
    print('M')
    print(M)
    h, w = src.shape
    dh, dw = src.shape
    dst = np.zeros((h, w))
    addc, addr = 0, 0

    if fit:
        P_test = np.array([[0], [0], [1]])
        xy_00 = np.dot(M, P_test)
        P_test = np.array([[w-1], [0], [1]])
        xy_w0 = np.dot(M, P_test)
        P_test = np.array([[0], [h-1], [1]])
        xy_0h = np.dot(M, P_test)
        P_test = np.array([[w-1], [h-1], [1]])
        xy_wh = np.dot(M, P_test)

        addc = int(np.ceil(min(xy_00[0], xy_0h[0])))
        addr = int(np.ceil(min(xy_00[1], xy_w0[1])))
        dw = int(np.ceil(max(xy_w0[0], xy_wh[0]) - addc))+1
        dh = int(np.ceil(max(xy_0h[1], xy_wh[1]) - addr))+1
        dst = np.zeros((dh, dw))

    N = np.zeros(dst.shape)

    for row in range(h):
        for col in range(w):
            P = np.array([
                [col],
                [row],
                [1]
            ])

            P_dst = np.dot(M, P)
            dst_col = P_dst[0][0]-addc
            dst_row = P_dst[1][0]-addr

            if dst_col >= dw:
                dst_col = dw-1
            elif dst_col < 0:
                dst_col = 0

            if dst_row >= dh:
                dst_row = dh-1
            elif dst_row < 0:
                dst_row = 0

            dst_col_right = int(np.ceil(dst_col))
            dst_col_left = int(dst_col)
            dst_row_bottom = int(np.ceil(dst_row))
            dst_row_top = int(dst_row)

            if dst_col_right >= dw:
                dst_col_right = dst_col_left

            if dst_row_bottom >= dh:
                dst_row_bottom = dst_row_top

            dst[dst_row_top, dst_col_left] += src[row, col]
            N[dst_row_top, dst_col_left] += 1

            if dst_col_right != dst_col_left:
                dst[dst_row_top, dst_col_right] += src[row, col]
                N[dst_row_top, dst_col_right] += 1

            if dst_row_bottom != dst_row_top:
                dst[dst_row_bottom, dst_col_left] += src[row, col]
                N[dst_row_bottom, dst_col_left] += 1

            if dst_col_right != dst_col_left and dst_row_bottom != dst_row_top:
                dst[dst_row_bottom, dst_col_right] += src[row, col]
                N[dst_row_bottom, dst_col_right] += 1

    dst = np.round(dst / (N + 1E-6))
    dst = dst.astype(np.uint8)
    return dst


def backward(src, M, fit=False):
    print('< backward >')
    print('M')
    print(M)
    dst = np.zeros((src.shape))
    dh, dw = dst.shape
    h, w = src.shape
    M_inv = np.linalg.inv(M)

    addc, addr = 0, 0

    if fit:
        P_test = np.array([[0], [0], [1]])
        xy_00 = np.dot(M, P_test)
        P_test = np.array([[w-1], [0], [1]])
        xy_w0 = np.dot(M, P_test)
        P_test = np.array([[0], [h-1], [1]])
        xy_0h = np.dot(M, P_test)
        P_test = np.array([[w-1], [h-1], [1]])
        xy_wh = np.dot(M, P_test)

        addc = int(np.ceil(min(xy_00[0], xy_0h[0])))
        addr = int(np.ceil(min(xy_00[1], xy_w0[1])))
        dw = int(np.ceil(max(xy_w0[0], xy_wh[0]) - addc))+1
        dh = int(np.ceil(max(xy_0h[1], xy_wh[1]) - addr))+1
        dst = np.zeros((dh, dw))

    for row in range(dh):
        for col in range(dw):
            P_dst = np.array([
                [col+addc],
                [row+addr],
                [1]
            ])

            P = np.dot(M_inv, P_dst)
            src_col = P[0, 0]
            src_row = P[1, 0]

            if src_col < 0 or src_row < 0:
                continue

            src_col_right = int(np.ceil(src_col))
            src_col_left = int(src_col)

            src_row_bottom = int(np.ceil(src_row))
            src_row_top = int(src_row)

            if src_col_right >= w or src_row_bottom >= h:
                continue

            s = src_col - src_col_left
            t = src_row - src_row_top

            intensity = (1-s) * (1-t) * src[src_row_top, src_col_left] \
                        + s * (1-t) * src[src_row_top, src_col_right] \
                        + (1-s) * t * src[src_row_bottom, src_col_right] \
                        + s * t * src[src_row_bottom, src_col_right]

            dst[row, col] = intensity

    dst = dst.astype(np.uint8)
    return dst


def main():
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)

    M_tr = np.array([
        [1, 0, -30],
        [0, 1, 50],
        [0, 0, 1]
    ])

    M_sc = np.array([
        [0.5, 0, 0],
        [0, 0.5, 0],
        [0, 0, 1]
    ])

    degree = -20
    M_ro = np.array([
        [np.cos(np.deg2rad(degree)), -np.sin(np.deg2rad(degree)), 0],
        [np.sin(np.deg2rad(degree)), np.cos(np.deg2rad(degree)), 0],
        [0, 0, 1]
    ])

    M_sh = np.array([
        [1, 0.2, 0],
        [0.2, 1, 0],
        [0, 0, 1]
    ])

    M = np.dot(M_sh,np.dot(M_sc,np.dot(M_tr,M_ro)))

    fit = True

    dst_for = forward(src, M, fit=fit)
    dst_for2 = forward(dst_for, np.linalg.inv(M), fit=fit)

    dst_back = backward(src, M, fit=fit)
    dst_back2 = backward(dst_back, np.linalg.inv(M), fit=fit)

    cv2.imshow('original', src)
    cv2.imshow('forward2', dst_for2)
    cv2.imshow('backward2', dst_back2)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ =='__main__':
    main()