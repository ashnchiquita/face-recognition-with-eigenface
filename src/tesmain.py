import configImages as ci
import numpy as np
import cv2

pathfolder = "../tesimg"
filesList = ci.filesInsideFolder(pathfolder,[])

# x untransposed
x = np.ndarray.flatten(ci.readImage(filesList[0]))

for i in range (1, len(filesList)):
    img = ci.readImage(filesList[i])
    img = np.ndarray.flatten(img)
    x = np.vstack((x, img))

# x transposed (vector)
# x = x.T
avgFace = np.mean(x, axis = 0).reshape(1, x.shape[1])
#berhasil
#muka = avgFace.reshape(256,256)
#cv2.imwrite("tes.jpg",muka)
A = x - np.tile(avgFace,(x.shape[0],1))

# kovarian
cov = A @ A.T

eigValCov, eigVecCov = np.linalg.eig(cov)
idx = eigValCov.argsort()[::-1]
eigValCov = eigValCov[idx]
eigVecCov = eigVecCov[:,idx]

# eigval Cov = eigval C'
# eigvec C' = A x eigvec Cov
# ERROR WARNING : FLOATING POINT NEGATIF, KASIH TOLERANSI BUAT JADI NOL. secara teoretis gabisa nilai eigvalnya negatif, tapi ada yg negatif & kecil bgt

# kalo mo lbh improve performance, itung rank nya di sini dulu baru perhitungannya dipotong, rank <= eigVecCov.shape[1]
eigVecC = (A.T @ eigVecCov[:,0])
rank = 1600
for i in range(1,eigVecCov.shape[1]): #for i in range(1,rank)
    #b[:,0] = i @ b[:,1]
    #print(eigVecCov[:,i])
    eigVecC = np.vstack((eigVecC,(A.T @ eigVecCov[:,i])))
eigVecC = eigVecC.T

eigVecC = eigVecC[:eigVecC.shape[1],:]

# w adalah matriks koefisien eigenface
w = []
for i in range(x.shape[0]):
    wi = np.linalg.inv(eigVecC) @ A[:,i]
    w.append(wi)
    
pathUnknown = "tesImg.jpg"
y = ci.readImage(pathUnknown)
print(y)
y = np.ndarray.flatten(y)
print(y)
y = y.reshape(1,x.shape[1])
print(y, avgFace)
aUnknown = np.subtract(y,avgFace).T[:x.shape[0],:]
print(eigVecC.shape,aUnknown.shape)

omega = np.linalg.inv(eigVecC) @ aUnknown

minDist = 0
closestImgIdx = -1
for i in range(len(filesList)):
    euDist = np.linalg.norm(np.subtract(w[i],omega))
    print(f"eudist ke-{i + 1}: {euDist}")
    if (euDist < minDist):
        minDist = euDist
        closestImgIdx = i
        
print(f"Path gambar terdekat: {filesList[i]}")