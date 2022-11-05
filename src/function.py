import numpy as np

# selisih(matrix, mean) -> matrix
def selisihMeanMat(matrix, mean):
# matrix: training image ke-i
# mean  : nilai mean dari seluruh himpunan matriks S
    return np.subtract(matrix, mean)

# eigenface(himpunan selisih matrix, eigenvector) -> matrix eigenface
def eigenfaceMat(selisih, eigenvector):
# S             : himpunan matriks selisih training image dengan mean
# eigenvector   : vektor eigen ke-i dari matriks kovarian
    miu = np.multiply(eigenvector, selisih[0])
    for i in range(1,len(selisih)):
        miu = np.add(miu, np.multiply(eigenvector, selisih[i]))
    return miu