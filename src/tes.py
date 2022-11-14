import function as fc

pathfolder = "realTes"
pathUnknown = "siapaCik1.jpg"

closestPath, percentage = fc.faceRecog(pathfolder, pathUnknown)

print(closestPath)