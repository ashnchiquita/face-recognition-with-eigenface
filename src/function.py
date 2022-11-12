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

def upper_triangle(matrix, Q): # Q is obtained from the decomposition of the matrix using the Gram-Schmidt procedure
    matrix_output = [[0 for j in range(len(matrix))] for i in range(len(matrix))]
    matrix = np.transpose(matrix)
    Q = np.transpose(Q)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if (i <= j):
                matrix_output[i][j] = np.dot(Q[i], matrix[j])
    return np.matrix(matrix_output)

# Finding Eigenvalue and Eigenvector

def eigen(matrix): # Precondition: the input matrix has to be symmetric
    eigenval = [0 for i in range(len(matrix))]
    eigenvector = np.identity(len(matrix))
    for i in range(100):
        Q = orthogonal_matrix(matrix)
        eigenvector = np.matmul(eigenvector, Q)
        matrix = np.transpose(Q) @ matrix @ Q
    
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if (i == j):
                eigenval[i] = matrix[i][j]

    return eigenval, eigenvector