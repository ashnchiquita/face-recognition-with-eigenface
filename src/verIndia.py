import configImages as ci
import numpy as np

def normalize(vec):
    length = np.linalg.norm(vec)
    if (length != 0):
        vec = vec / length
    return vec

pathfolder = "../test"
filesList = ci.filesInsideFolder(pathfolder,[])
nData = len(filesList)
imSize = 256
imVecSize = imSize ** 2

# xT : nData x 65535, 1 image 1 baris
xT = np.ndarray.flatten(ci.readImage(filesList[0]))
for i in range (1, nData):
    img = ci.readImage(filesList[i])
    img = np.ndarray.flatten(img)
    xT = np.vstack((xT, img))

# mean : 1 x 65535
mean = np.mean(xT, axis = 0).reshape(1, imVecSize)

# aT : nData x 65535, 1 image 1 baris
aT = xT - np.tile(mean,(nData,1))

# simpCov : nData x nData
simpCov = aT @ aT.T

# choose rank
k = nData

# eigVal : nData
# eigVecSimpCov : nData x nData
eigVal, eigVecSimpCov = np.linalg.eig(simpCov)
idx = eigVal.argsort()[::-1]
eigVal = eigVal[idx]
eigVecSimpCov = eigVecSimpCov[:,idx]

# eigVecCovT : nData x 65536
eigVecCovT = normalize(aT.T @ eigVecSimpCov[:,0])
for i in range(1,k): #for i in range(1,rank)
    eigVecCovT = np.vstack((eigVecCovT, normalize(aT.T @ eigVecSimpCov[:,i])))

u = eigVecCovT.T


# calculate each w
currA = aT[0,:].reshape(imVecSize,1)
w = (eigVecCovT[0,:].reshape(1,imVecSize)) @ currA
for j in range(1,k):
    w = np.vstack((w,(eigVecCovT[j,:].reshape(1,imVecSize)) @ currA))
# new image data
xT[0,:] = np.add(mean, (u @ w).reshape(1, imVecSize))

omega = w.reshape(1,nData)
# for each image
for i in range(1,nData):
    # calculate each w
    currA = aT[i,:].reshape(imVecSize,1)
    w = (eigVecCovT[0,:].reshape(1,imVecSize)) @ currA
    for j in range(1,k):
        w = np.vstack((w,(eigVecCovT[j,:].reshape(1,imVecSize)) @ currA))
    # new image data
    xT[i,:] = np.add(mean, (u @ w).reshape(1, imVecSize))
    omega = np.vstack((omega, w.reshape(1,nData)))


pathUnknown = "tesImg.jpg"
yT = np.ndarray.flatten(ci.readImage(pathUnknown)).reshape(1,imVecSize)

ayT = yT - mean
currAY = ayT.reshape(imVecSize,1)
wNew = (eigVecCovT[0,:].reshape(1,imVecSize)) @ currAY
for j in range(1,k):
    wNew = np.vstack((wNew,(eigVecCovT[j,:].reshape(1,imVecSize)) @ currAY))

yT[0,:] = np.add(mean, (u @ wNew).reshape(1, imVecSize))

omegaNew = np.squeeze(np.asarray(wNew.reshape(1,nData)))


minDist = np.linalg.norm(np.subtract(omegaNew,omega[0,:]))
print(f"Euclidean Distance {filesList[0]}: {minDist}")
closestImgIdx = 0
for i in range(1,nData):
    euDist = np.linalg.norm(np.subtract(omegaNew,omega[i,:]))
    print(f"Euclidean Distance {filesList[i]}: {euDist}")
    if (euDist < minDist):
        minDist = euDist
        closestImgIdx = i
        
print(f"Path gambar terdekat dengan {pathUnknown}: {filesList[closestImgIdx]}")