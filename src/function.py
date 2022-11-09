import numpy as np
            
# mean(list_of_matrix) -> matrix


def mean(list_of_matrix):
    sum = [[0 for i in range(256)] for j in range(256)]
    for img in list_of_matrix:
        sum = np.add(sum, img) # Menambahkan semua matriks
    mean = np.divide(sum, len(list_of_matrix))
    return mean

# selisih(matrix, mean) -> matrix


def selisihMeanMat(matrix, mean):
    # matrix: training image ke-i
    # mean  : nilai mean dari seluruh himpunan matriks S
    return np.subtract(matrix, mean)

# multiplyTranspose(matrix) -> matrix
def multiplyTranspose(matrix): # Melakukan perkalian matrix
    transpose = np.transpose(matrix)
    result = np.matmul(matrix, transpose)
    return result

# covarian(listOfPhi) -> matriks


def covarian(listOfPhi):  # Menghitung kovarian
    sum = multiplyTranspose(listOfPhi[0])
    for i in range(1, len(listOfPhi)):
        sum += multiplyTranspose(listOfPhi[i])
    cov = np.divide(sum, len(listOfPhi))
    return cov

# minorMatrix(matrix, row, col) -> list


def minorMatrix(matrix, row, col):  # Mengembalikan matriks minor
    array = matrix[0]
    minor = [[0 for j in range(len(array)-1)]
             for i in range(len(array)-1)]  # Inisialisasi matriks
    for i in range(0, len(array)):
        for j in range(0, len(array)):
            if (i < row):
                if (j > col):
                    minor[i][j-1] = matrix[i][j]
                elif (j < col):
                    minor[i][j] = matrix[i][j]
            if (i > row):
                if (j > col):
                    minor[i-1][j-1] = matrix[i][j]
                elif (j < col):
                    minor[i-1][j] = matrix[i][j]
    return minor

# determinanCofactor(matrix_covarian) -> list berisi koefisien-koefisien dari persamaan yang akan diselesaikan untuk menemukan nilai eigen


# Menggunakan metode kofaktor untuk menemukan determinannya
def determinanCofactor(matrix_covarian):
    array = matrix_covarian[0]
    mTemp = [[0 for j in range(len(array)-1)] for i in range(len(array)-1)]
    det = 0
    sign = 1
    if (len(array) != 1):
        for i in range(0, len(array)):
            mTemp = minorMatrix(matrix_covarian, 0, i)
            if (type(matrix_covarian[0][i]) != list):
                det += (determinanCofactor(mTemp)*matrix_covarian[0][i]*sign)
                sign *= (-1)
            else:
                if (type(determinanCofactor(mTemp)) != list):
                    det += matrix_covarian[0][i]*determinanCofactor(mTemp)*sign
                else:
                    det += ((np.polymul(determinanCofactor(mTemp),
                            matrix_covarian[0][i]))*(sign))
                sign *= (-1)
    else:
        det = matrix_covarian[0][0]
    return det

# eigenvalues(matrix_covarian) -> list berisi nilai eigen


def eigenval(matrix_covarian):  # Mengembalikan list berisi nilai eigen
    array = matrix_covarian[0]
    for i in range(len(array)):
        for j in range(len(array)):
            if (i == j):
                matrix_covarian[i][j] = [(-1)*matrix_covarian[i][j]]
                matrix_covarian[i][j].insert(0, 1)
            else:
                matrix_covarian[i][j] = matrix_covarian[i][j]*(-1)
    mTemp = [[0 for j in range(len(array)-1)] for i in range(len(array)-1)]
    det = 0
    sign = 1
    if (len(array) != 1):
        for i in range(0, len(array)):
            mTemp = minorMatrix(matrix_covarian, 0, i)
            if (type(matrix_covarian[0][i]) != list):
                det += (determinanCofactor(mTemp)*matrix_covarian[0][i]*sign)
                sign *= (-1)
            else:
                if (type(determinanCofactor(mTemp)) != list):
                    det += matrix_covarian[0][i]*determinanCofactor(mTemp)*sign
                else:
                    det += ((np.polymul(determinanCofactor(mTemp),
                            matrix_covarian[0][i]))*(sign))
                sign *= (-1)
    else:
        det = matrix_covarian[0][0]
    return np.roots(det)

# eigenface(himpunan selisih matrix, eigenvector) -> matrix eigenface


def eigenfaceMat(selisih, eigenvector):
    # S             : himpunan matriks selisih training image dengan mean
    # eigenvector   : vektor eigen ke-i dari matriks kovarian
    miu = np.multiply(eigenvector, selisih[0])
    for i in range(1, len(selisih)):
        miu = np.add(miu, np.multiply(eigenvector, selisih[i]))
    return miu

def euclidean_norm(vector):
    total = 0
    for a in vector:
        total += (a**2)
    total = total**(1/2)
    return total

# QR Decomposition Using Gram-Schmidt Procedure

def orthogonal_matrix(matrix):
    matrix = np.transpose(matrix)
    result = []
    for vector in matrix:
        sum = [0 for i in range(len(matrix))]
        for a in result:
            sum = np.add(sum, np.dot(vector, a)*a)
        a = np.subtract(vector, sum)
        result.append(a/euclidean_norm(a))
    result = np.transpose(result)
    return np.array(result)

def upper_triangle(matrix, e):
    matrix_output = [[0 for j in range(len(matrix))] for i in range(len(matrix))]
    matrix = np.transpose(matrix)
    e = np.transpose(e)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if (i <= j):
                matrix_output[i][j] = np.dot(e[i], matrix[j])
    return np.matrix(matrix_output)