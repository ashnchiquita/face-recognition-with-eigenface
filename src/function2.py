#biar g conflicting hehe
import numpy as np

# TAHAP PENGENALAN WAJAH
def eigenfaceNew(eigenVecs, matNew, meanMat):
# Mengembalikan nilai eigenface dari testing image
# INI AGAK SUS GES sori belom yakin maksud rumusnya gimana soalnya di referensi beda
    mergedEigenVecs = eigenVecs[0]
    for i in range(1, len(eigenVecs)):
        mergedEigenVecs = np.hstack(mergedEigenVecs, eigenVecs[i])
    return np.multiply(mergedEigenVecs, np.subtract(matNew, meanMat))

def euclidDist(eigenfaceNew, eigenfaceTraining):
# Mengembalikan euclidean distance antara eigenfaceNew dan 1 eigenfaceTraining
    vec = np.subtract(eigenfaceNew, eigenfaceTraining)
    return np.linalg.norm(vec)