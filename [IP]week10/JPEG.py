import numpy as np
import cv2


def C(w, n = 8):
    if w == 0:
        return (1/n)**0.5
    else:
        return (2/n)**0.5


def Quantization_Luminance():
    luminance = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    return luminance


def img2block(src, n=8):
    v, u = src.shape
    blocks = []
    for row in range(v//n):
        for col in range(u//n):
            blocks.append(src[row*n:(row+1)*n, col*n:(col+1)*n])

    return np.array(blocks)


def DCT(block, n=8):
    dst = np.zeros(block.shape)
    y, x = np.mgrid[0:n, 0:n]

    for v_ in range(n):
        for u_ in range(n):
            tmp = block * np.cos(((2*x+1)*u_*np.pi)/(2*n)) * np.cos(((2*y+1)*v_*np.pi)/(2*n))
            dst[v_][u_] = C(v_, n) * C(u_, n) * np.sum(tmp)

    return np.round(dst)


def my_zigzag_scanning(block, mode='encoding', block_size=8):
    cell = np.array([0, 0])
    if mode == 'encoding':
        zigzaged = []
        addlist = [[-1, 1], [1, -1]]
        changelist = [[0, 1], [1, 0]]

        for i in range(block_size):
            add = addlist[i % 2]
            zigzaged.append(block[cell[0]][cell[1]])
            for j in range(i):
                cell = np.add(cell, add)
                zigzaged.append(block[cell[0]][cell[1]])
            change = changelist[i % 2]
            cell = np.add(cell, change)

        cell = np.add(cell, [-1, 1])

        for i in range(1, block_size):
            add = addlist[1 - i % 2]
            zigzaged.append(block[cell[0]][cell[1]])
            for j in range(block_size - i - 1):
                cell = np.add(cell, add)
                zigzaged.append(block[cell[0]][cell[1]])
            change = changelist[i % 2]
            cell = np.add(cell, change)

        for i in range(block_size * block_size - 1, 0, -1):
            if zigzaged[i] != 0:
                break
            else:
                zigzaged.pop(i)
        zigzaged.append('EOB')
    else:
        zigzaged = [[0 for _ in range(block_size)] for _ in range(block_size)]
        addlist = [[-1, 1], [1, -1]]
        changelist = [[0, 1], [1, 0]]
        count = 0
        for i in range(block_size):
            if block[count] == 'EOB':
                break
            add = addlist[i % 2]
            zigzaged[cell[0]][cell[1]] = block[count]
            count += 1
            for j in range(i):
                if block[count] == 'EOB':
                    break
                cell = np.add(cell, add)
                zigzaged[cell[0]][cell[1]] = block[count]
                count += 1
            change = changelist[i % 2]
            cell = np.add(cell, change)

        cell = np.add(cell, [-1, 1])

        for i in range(1, block_size):
            if block[count] == 'EOB':
                break
            add = addlist[1 - i % 2]
            zigzaged[cell[0]][cell[1]] = block[count]
            count += 1
            for j in range(block_size - i - 1):
                if block[count] == 'EOB':
                    break
                cell = np.add(cell, add)
                zigzaged[cell[0]][cell[1]] = block[count]
                count += 1
            change = changelist[i % 2]
            cell = np.add(cell, change)

    return zigzaged


def DCT_inv(block, n = 8):
    dst = np.zeros(block.shape)
    v, u = dst.shape
    y, x = np.mgrid[0:u, 0:v]

    for i in range(n):
        for j in range(n):
            block[i][j] = C(i, n) * C(j, n) * block[i][j]

    for v_ in range(n):
        for u_ in range(n):
            tmp = block * np.cos(((2 * u_ + 1) * x * np.pi) / (2 * n)) * np.cos(((2 * v_ + 1) * y * np.pi) / (2 * n))
            dst[v_][u_] = np.sum(tmp)

    return np.round(dst)


def block2img(blocks, src_shape, n = 8):
    dst = np.zeros(src_shape)
    v, u = map(int, np.divide(src_shape,(n,n)))

    count = 0
    for row in range(v):
        for col in range(u):
            dst[row*n:(row+1)*n, col*n:(col+1)*n] = blocks[count]
            count+=1
    return dst.astype(np.uint8)


def Encoding(src, n=8):
    print('<start Encoding>')
    blocks = img2block(src, n=n)
    blocks -= 128

    blocks_dct = []
    for block in blocks:
        blocks_dct.append(DCT(block, n=n))
    blocks_dct = np.array(blocks_dct)

    Q = Quantization_Luminance()
    QnT = np.round(blocks_dct / Q)

    zz = []
    for i in range(len(QnT)):
        zz.append(my_zigzag_scanning(QnT[i]))

    return zz, src.shape


def Decoding(zigzag, src_shape, n=8):
    print('<start Decoding>')

    blocks = []
    for i in range(len(zigzag)):
        blocks.append(my_zigzag_scanning(zigzag[i], mode='decoding', block_size=n))
    blocks = np.array(blocks)

    Q = Quantization_Luminance()
    blocks = blocks * Q

    blocks_idct = []
    for block in blocks:
        blocks_idct.append(DCT_inv(block, n=n))
    blocks_idct = np.array(blocks_idct)

    blocks_idct += 128
    dst = block2img(blocks_idct, src_shape=src_shape, n=n)

    return dst


def main():
    src = cv2.imread('../imgs/Lena.png', cv2.IMREAD_GRAYSCALE)
    comp, src_shape = Encoding(src, n=8)

    recover_img = Decoding(comp, src_shape, n=8)

    cv2.imshow('recover img', recover_img)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
