import cv2
import os
from PIL import Image
import time


def readImage(pathFile):
    # Mengubah 1 image dengan path pathFile menjadi Mat, grayscale, dan berukuran 256 x 256
    # pathFile : string path dari file
    img = cv2.imread(pathFile, 0)
    img = cv2.resize(img, (256, 256))
    return img


def convertFrame(frame):
    # Mengubah 1 frame menjadi grayscale, dan berukuran 256 x 256
    # frame : Mat dari sebuah image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (256, 256))
    return img


def resizeImage(source):
    resultImage = cv2.imread(source)
    resultImage = cv2.cvtColor(resultImage, cv2.COLOR_BGR2RGB)
    resultImage = cv2.resize(resultImage, (256, 256))
    return Image.fromarray(resultImage)


def takePhoto():
    # Mengambil foto dari webcam, mengembalikan foto tersebut (tanpa diconvert)
    # Jika gagal mengembalikan None
    global cam
    global img
    start = time.time()

    cam_port = 0
    cam = cv2.VideoCapture(cam_port, cv2.CAP_DSHOW)

    while (True):
        result, img = cam.read()
        cv2.imshow('Camera', img)

        if cv2.waitKey(1) & (time.time()-start > 3):
            break

    cam.release()
    cv2.destroyAllWindows()

    if result:
        return img
    else:
        return None


def readFolderImages(folderName):
    # Mengembalikan list berisi Mat dari image yang ada di setiap folder
    matrixList = []
    files = filesInsideFolder(folderName, [])
    count = 0
    for imgPath in files:
        img = readImage(imgPath)
        matrixList.append(img)
        count += 1
        print(f"Converting images ({count}/{len(files)})")
    return matrixList


def filesInsideFolder(folderName, listFiles):
    # Mengembalikan list berisi path-path dari setiap file images yang ada di folder
    # Mengabaikan file dengan format selain image yang disupport oleh OpenCV
    # folderName : string nama folder
    # listFiles : list of string, untuk nyimpen path filesnya (inisialisasinya pake [] aja)
    for filename in os.listdir(folderName):
        filepath = os.path.join(folderName, filename)
        if (os.path.isfile(filepath)):
            if (filepath.endswith(".png") or filepath.endswith(".jpg") or filepath.endswith(".jpeg") or filepath.endswith(".jpe") or filepath.endswith(".jp2") or filepath.endswith(".bmp") or filepath.endswith(".pbm") or filepath.endswith(".pgm") or filepath.endswith(".ppm") or filepath.endswith(".sr") or filepath.endswith(".ras") or filepath.endswith(".tiff") or filepath.endswith(".tif")):
                listFiles.append(filepath)
        elif (os.path.isdir(filepath)):
            filesInsideFolder(filepath, listFiles)
    return listFiles


def showImg(windowName, img):
    # Menampilkan image
    # windowName : string nama window
    # img : mat gambar yang bakal ditampilin
    cv2.imshow(windowName, img)
    return


def writeImg(pathString, img):
    # Menyimpan image di path pathString
    # pathString : string path tempat menyimpan image
    # img : mat gambar yang bakal ditampilin
    cv2.imwrite(pathString, img)
    return

# Testing / ref kode di main

# img = takePhoto()
# if (img is not None):
#     cv2.imwrite('wokwok.png', img)
# else:
#     print("gagal")

# pathfolder = input("nama folder: ")
# print(filesInsideFolder(pathfolder, []))
# matList = readFolderImages(pathfolder)
# num = 0
# for images in matList:
#     cv2.imwrite((str(num) + '.png'), images)
#     print('berhasil')
#     num += 1
