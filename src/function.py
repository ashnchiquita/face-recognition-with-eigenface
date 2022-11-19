import numpy as np
import configImages as ci


def normalize(vec):
    length = np.linalg.norm(vec)
    if (length != 0):
        vec = vec / length
    return vec


def mean(list_of_matrix):
    sum = [[0 for i in range(256)] for j in range(256)]
    for img in list_of_matrix:
        sum = np.add(sum, img)  # Menambahkan semua matriks
    mean = np.divide(sum, len(list_of_matrix))
    return mean

# selisih(matrix, mean) -> matrix


def selisihMeanMat(matrix, mean):
    # matrix: training image ke-i
    # mean  : nilai mean dari seluruh himpunan matriks S
    return np.subtract(matrix, mean)

# multiplyTranspose(matrix) -> matrix


def multiplyTranspose(matrix):  # Melakukan perkalian matrix
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


# Q is obtained from the decomposition of the matrix using the Gram-Schmidt procedure
def upper_triangle(matrix, Q):
    matrix_output = [[0 for j in range(len(matrix))]
                     for i in range(len(matrix))]
    matrix = np.transpose(matrix)
    Q = np.transpose(Q)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if (i <= j):
                matrix_output[i][j] = np.dot(Q[i], matrix[j])
    return np.matrix(matrix_output)

# Finding Eigenvalue and Eigenvector


def eigen(matrix):  # Precondition: the input matrix has to be symmetric
    eigenval = [0 for i in range(len(matrix))]
    eigenvector = np.identity(len(matrix))
    for i in range(100):
        Q = orthogonal_matrix(matrix)
        eigenvector = np.matmul(eigenvector, Q)
        matrix = Q.T @ matrix @ Q

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if (i == j):
                eigenval[i] = matrix[i][j]

    return eigenval, eigenvector


def testImgCam(rawTestImgMat):
    # rawTestImgMat masih raw (hasil takePhoto()), diubah jadi 256 x 256 grayscale
    testImgMat = ci.convertFrame(rawTestImgMat)
    return testImgMat


def testImgFile(pathfile):
    # bikin jadi 256 x 256 grayscale dari file
    testImgMat = ci.readImage(pathfile)
    return testImgMat


def faceRecog(pathfolder, testImgMat):
    filesList = ci.filesInsideFolder(pathfolder, [])
    nData = len(filesList)
    imSize = 256
    imVecSize = imSize ** 2

    # xT : nData x 65535, 1 image 1 baris
    xT = np.ndarray.flatten(ci.readImage(filesList[0]))
    for i in range(1, nData):
        img = ci.readImage(filesList[i])
        img = np.ndarray.flatten(img)
        xT = np.vstack((xT, img))

    # mean : 1 x 65535
    mean = np.mean(xT, axis=0).reshape(1, imVecSize)

    # aT : nData x 65535, 1 image 1 baris
    aT = xT - np.tile(mean, (nData, 1))

    # simpCov : nData x nData
    simpCov = aT @ aT.T

    # choose rank
    k = nData

    # eigVal : nData
    # eigVecSimpCov : nData x nData
    eigVal, eigVecSimpCov = eigen(simpCov)
    eigVal = np.array(eigVal, dtype=np.float32)
    idx = eigVal.argsort()[::-1]
    eigVal = eigVal[idx]
    eigVecSimpCov = eigVecSimpCov[:, idx]

    # eigVecCovT : nData x 65536
    eigVecCovT = normalize(aT.T @ eigVecSimpCov[:, 0])
    for i in range(1, k):  # for i in range(1,rank)
        eigVecCovT = np.vstack(
            (eigVecCovT, normalize(aT.T @ eigVecSimpCov[:, i])))

    u = eigVecCovT.T

    # calculate each w
    currA = aT[0, :].reshape(imVecSize, 1)
    w = (eigVecCovT[0, :].reshape(1, imVecSize)) @ currA
    for j in range(1, k):
        w = np.vstack((w, (eigVecCovT[j, :].reshape(1, imVecSize)) @ currA))

    # new image data (reconstruction)
    xT[0, :] = np.add(mean, (u @ w).reshape(1, imVecSize))

    omega = w.reshape(1, nData)
    # for each image
    for i in range(1, nData):
        # calculate each w
        currA = aT[i, :].reshape(imVecSize, 1)
        w = (eigVecCovT[0, :].reshape(1, imVecSize)) @ currA
        for j in range(1, k):
            w = np.vstack(
                (w, (eigVecCovT[j, :].reshape(1, imVecSize)) @ currA))
        # new image data
        xT[i, :] = np.add(mean, (u @ w).reshape(1, imVecSize))
        omega = np.vstack((omega, w.reshape(1, nData)))

    yT = np.ndarray.flatten(testImgMat).reshape(1, imVecSize)

    ayT = yT - mean
    currAY = ayT.reshape(imVecSize, 1)
    wNew = (eigVecCovT[0, :].reshape(1, imVecSize)) @ currAY
    for j in range(1, k):
        wNew = np.vstack(
            (wNew, (eigVecCovT[j, :].reshape(1, imVecSize)) @ currAY))

    yT[0, :] = np.add(mean, (u @ wNew).reshape(1, imVecSize))

    omegaNew = np.squeeze(np.asarray(wNew.reshape(1, nData)))

    minDist = np.linalg.norm(np.subtract(omegaNew, omega[0, :]))
    maxDist = minDist
    closestImgIdx = 0
    for i in range(1, nData):
        euDist = np.linalg.norm(np.subtract(omegaNew, omega[i, :]))
        if (euDist < minDist):
            minDist = euDist
            closestImgIdx = i
        if (euDist > maxDist):
            maxDist = euDist

    threshold = 0.9 * maxDist
    recognized = (minDist < threshold)

    print(minDist)
    print(maxDist)

    percentage = 100 - ((minDist/maxDist)*100)
    return filesList[closestImgIdx], recognized, percentage
    # return path terdekat, percentage (nyusul sepertinya saya butuh riset lebih jauh di sini)
