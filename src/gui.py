from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2
import configImages
from timeit import default_timer as timer
import function

usingPath = True

# initialize root and title
main = Tk()
main.title('Face Recognition with EigenFace')
main.iconbitmap('external/logo.ico')
main.configure(bg='#FFEADF')

# frame of all
root = LabelFrame(main, relief=FLAT, bg='#FFEADF')
root.pack(padx=50)

# Header
headerFrame = LabelFrame(root, relief=FLAT)
headerFrame.pack()
headerFrame.configure(bg='#FFEADF')
header = Label(
    headerFrame, text='Face Recognition with EigenFace', font=("Montserrat", 20, "bold"), bg='#FFEADF').pack(pady=20)

# Image Description
imageFrame = LabelFrame(root, bg='#FEB20E')
imageFrame.pack()
desc1 = Label(imageFrame, text='Test Image', font=("Arial", 15, "bold"), bg='#FEB20E').grid(
    row=0, column=0, padx=10, pady=(10, 0))
desc2 = Label(imageFrame, text='Closest Result', font=("Arial", 15, "bold"), bg='#FEB20E').grid(
    row=0, column=1, padx=10, pady=(10, 0))

# Image Container
testResized = configImages.resizeImage('external/ayang.jpg')
testImage = ImageTk.PhotoImage(image=testResized)
testLabel = Label(imageFrame, image=testImage)
testLabel.grid(row=1, column=0, padx=10, pady=10)

resultImage = configImages.resizeImage(
    'C:/Users/ASUS/Pictures/BG/bae2\wp4891275.jpg')
closestResult = ImageTk.PhotoImage(image=resultImage)
resultLabel = Label(imageFrame, image=closestResult)
resultLabel.grid(row=1, column=1, padx=10, pady=10)

# Status
status = 'Program is ready...'

statusFrame = LabelFrame(root, width=570, height=22, relief=FLAT, bg='#FFEADF')
statusFrame.grid_propagate(0)
statusFrame.pack(pady=(10, 0))

statusLabel = Label(statusFrame, text=f"{status}", anchor='e', bg='#FFEADF', font=(
    "Arial", 10, "bold"),)
statusLabel.grid(
    column=0, row=0, sticky='w')


# Source file
path = 'None'

srcFrame = LabelFrame(root, width=570, height=22, relief=FLAT, bg='#FFEADF')
srcFrame.grid_propagate(0)
srcFrame.pack(pady=(10, 0))

srcFrame.columnconfigure(0, weight=1, uniform='col')
srcFrame.columnconfigure(1, weight=5, uniform='col')

resLabel = Label(srcFrame, text=f"Result path : ", anchor='e', bg='#FFEADF', font=(
    "Arial", 10, "bold"))
resLabel.grid(
    column=0, row=0, sticky='w')

srcLabel = Label(srcFrame, text=f"{path}", anchor='e',
                 bg='#FFEADF', font=("Arial", 10, "bold"))
srcLabel.grid(column=1, row=0, sticky='w')


# Frame for desc
descFrame = LabelFrame(root, width=570, height=110, relief=FLAT, bg='#FFEADF')
descFrame.grid_propagate(0)
descFrame.pack(pady=20)

descFrame.columnconfigure(0, weight=1, uniform='col')
descFrame.columnconfigure(1, weight=3, uniform='col')
descFrame.columnconfigure(2, weight=1, uniform='col')


# Choose Test Image
filename = 'No Files Chosen'


def chooseTest():
    global filename
    global testImage
    global usingPath
    filename = filedialog.askopenfile(filetypes=[(
        'Image files', '*.jpg *.png *.jpeg *.jpe *.jp2 *.bmp *.pbm *.pgm *.ppm *.sr *.ras *.tiff *.tif')])
    if (filename):
        testDir.config(text=filename.name)
        testDir.update_idletasks()

        testResized = configImages.resizeImage(filename.name)
        testImage = ImageTk.PhotoImage(image=testResized)
        testLabel.config(image=testImage)
        testLabel.update_idletasks()

        usingPath = True


testButton = Button(descFrame, text='Choose Test Image',
                    command=chooseTest, width=16, height=2).grid(row=1, column=0, pady=(0, 10))
testDir = Label(descFrame, text=filename, font=(
    "Arial", 10, "bold"), bg='#FFEADF', anchor='e')
testDir.grid(row=1, column=1, sticky='w', pady=(0, 10), padx=(10, 5))

# Choose Dataset

foldername = 'No Folders Chosen'


def chooseDataset():
    global foldername
    foldername = filedialog.askdirectory()
    if foldername:
        dataDir.config(text=foldername)
        dataDir.update_idletasks()


dataButton = Button(descFrame, text='Choose Dataset',
                    command=chooseDataset, width=16, height=2).grid(row=2, column=0, pady=(10, 0))
dataDir = Label(descFrame, text=foldername, font=(
    "Arial", 10, "bold"), bg='#FFEADF', anchor='e')
dataDir.grid(row=2, column=1, sticky='w', pady=(10, 0), padx=(10, 5))

# Execute Recognize


def generate():
    global elapsedTime
    global resultLabel
    global resultImage
    global closestResult
    global ansPath

    if foldername != 'No Folders Chosen' and filename != 'No Files Chosen':

        start = timer()

        # process
        if usingPath:
            # kirim dalam bentuk path
            ansPath, recognized, percentage = function.faceRecog(
                foldername, function.testImgFile(filename.name))
        else:
            # kirim dalam bentuk array
            ansPath, recognized, percentage = function.faceRecog(
                foldername, function.testImgCam(imgCamToSend))

        # after processed
        if recognized:
            # change source path
            srcLabel.config(
                text=f'{ansPath}')
            srcLabel.update_idletasks()

            # change closest result image
            resultImage = configImages.resizeImage(ansPath)
            closestResult = ImageTk.PhotoImage(image=resultImage)
            resultLabel.config(image=closestResult)
            resultLabel.update_idletasks()

            # change status label
            statusLabel.config(
                text=f'Recognized! Matches {percentage} %', fg='#000000')
            statusLabel.update_idletasks()
        else:
            # change status label
            statusLabel.config(
                text=f'Fail to recognize!')
            statusLabel.update_idletasks()

        end = timer()
        elapsedTime = end-start

        timeLabel.config(text=f'Execution time : {elapsedTime:.2f} seconds')
        timeLabel.update_idletasks()

    else:
        statusLabel.config(
            text=f'Warning : Please choose test image or dataset', fg='#FF0000')
        statusLabel.update_idletasks()


generateButton = Button(descFrame, text='Generate', font=("Montserrat", 12, "bold"), bg='#1F307C', fg='#FFFFFF', width=10, command=generate).grid(
    row=1, column=2, padx=(5, 0), pady=5)

# Live Detect


def detectLive():
    global imgResized
    global imgCamToSend

    img = configImages.takePhoto()

    # imgCamToSend = configImages.convertFrame(img)
    imgCamToSend = img

    imgResized = cv2.resize(img, (256, 256))
    imgResized = cv2.cvtColor(imgResized, cv2.COLOR_BGR2RGB)

    return imgResized


def detect():
    global testImage
    global usingPath
    global imgCamera
    global filename

    imgCamera = detectLive()

    testDir.config(text="Test Image from Live Camera")
    testDir.update_idletasks()

    testImage = ImageTk.PhotoImage(image=Image.fromarray(imgCamera))
    testLabel.config(image=testImage)
    testLabel.update_idletasks()

    usingPath = False
    filename = ''


liveDetect = Button(descFrame, text='Live Detect', font=("Montserrat", 12, "bold"), bg='#1F307C', fg='#FFFFFF', width=10, command=detect).grid(
    row=2, column=2, padx=(5, 0), pady=5)

# Execution Time
exeFrame = LabelFrame(root, relief=FLAT, bg='#FFEADF')
exeFrame.pack(pady=(0, 20))

elapsedTime = 0.00

timeLabel = Label(
    exeFrame, text=f'Execution time : {elapsedTime} seconds', bg='#FFEADF')
timeLabel.grid(row=5, columnspan=4)

root.mainloop()
