import function as fc

pathfolder = "../test/TrainingImg/IF21"
pathUnknown = "../test/TestingImg/IF21/fiotest.jpg"

testImgMat = fc.testImgFile(pathUnknown)
closestPath, mirip, percentage = fc.faceRecog(pathfolder, testImgMat)

print(closestPath, percentage)