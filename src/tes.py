import function as fc

pathfolder = "../tempTrainingImg"
pathUnknown = "../testImg/siapaCik1(AdrianaLima).jpg"

testImgMat = fc.testImgFile(pathUnknown)
closestPath,mirip, percentage = fc.faceRecog(pathfolder, testImgMat)

print(closestPath)