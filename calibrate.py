import numpy as np

INF = 1e12


# -------------------------------------------------
# LOAD IMAGE
# -------------------------------------------------
def load_and_preprocess(path):
    with open(path, "rb") as f:
        header = f.read(512)
        img_data = np.fromfile(f, dtype=np.uint8)

    img = img_data.reshape((512, 512))
    binary = (img <= 128).astype(np.uint8)
    return binary, header


# -------------------------------------------------
# EXACT 1D EUCLIDEAN DT
# -------------------------------------------------
def edt_1d(f):
    n = len(f)
    d = np.zeros(n)
    v = np.zeros(n, dtype=int)
    z = np.zeros(n + 1)

    k = 0
    v[0] = 0
    z[0] = -INF
    z[1] = INF

    for q in range(1, n):
        s = ((f[q] + q*q) - (f[v[k]] + v[k]*v[k])) / (2*(q - v[k]))
        while s <= z[k]:
            k -= 1
            s = ((f[q] + q*q) - (f[v[k]] + v[k]*v[k])) / (2*(q - v[k]))
        k += 1
        v[k] = q
        z[k] = s
        z[k + 1] = INF

    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        d[q] = (q - v[k])**2 + f[v[k]]

    return d


# -------------------------------------------------
# EXACT 2D EUCLIDEAN DT
# -------------------------------------------------
def euclidean_dt(binary):
    H, W = binary.shape
    f = np.where(binary == 0, 0, INF)

    dt = np.zeros((H, W))

    # vertical pass
    for x in range(W):
        dt[:, x] = edt_1d(f[:, x])

    # horizontal pass
    for y in range(H):
        dt[y, :] = edt_1d(dt[y, :])

    return np.sqrt(dt)


# -------------------------------------------------
# EXTRACT MEDIAL AXIS
# -------------------------------------------------
def extract_medial_axis(dt):
    H, W = dt.shape
    M = np.zeros_like(dt)

    neighbors = [(-1,0),(1,0),(0,-1),(0,1),
                 (-1,-1),(-1,1),(1,-1),(1,1)]

    for y in range(H):
        for x in range(W):
            if dt[y, x] == 0:
                continue

            is_max = True
            for dy, dx in neighbors:
                ny, nx = y+dy, x+dx
                if 0 <= ny < H and 0 <= nx < W:
                    if dt[ny, nx] > dt[y, x]:
                        is_max = False
                        break

            if is_max:
                M[y, x] = dt[y, x]

    return M


# -------------------------------------------------
# LOSSLESS RECONSTRUCTION
# -------------------------------------------------
def reconstruct_from_medial_axis(M):
    H, W = M.shape

    f = np.where(M > 0, M**2, INF)

    dt = np.zeros((H, W))

    # vertical pass
    for x in range(W):
        dt[:, x] = edt_1d(f[:, x])

    # horizontal pass
    for y in range(H):
        dt[y, :] = edt_1d(dt[y, :])

    dt = np.sqrt(dt)

    return (dt > 0).astype(np.uint8)


# -------------------------------------------------
# MAIN
# -------------------------------------------------
if __name__ == "__main__":

    B, header = load_and_preprocess("comb.img")

    print("Computing DT...")
    DT = euclidean_dt(B)

    print("Extracting medial axis...")
    M = extract_medial_axis(DT)

    print("Reconstructing...")
    BR = reconstruct_from_medial_axis(M)

    error = np.sum(B != BR)
    print("Pixel error:", error)