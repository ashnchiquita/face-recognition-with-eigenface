import function as fc

pathfolder = "/Users/ashnchiquita/Documents/Tubes/Algeo2/Algeo02-21046/tempTrainingImg"
pathUnknown = "/Users/ashnchiquita/Documents/Tubes/Algeo2/Algeo02-21046/testImg/siapaCik1(AdrianaLima).jpg"

closestPath,mirip, percentage = fc.faceRecog(pathfolder, pathUnknown)

print(closestPath)