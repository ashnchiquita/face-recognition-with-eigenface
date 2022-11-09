from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog
import cv2
import configImages
from timeit import default_timer as timer

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
testResized = configImages.resizeImage('src/Mean.jpg')
testImage = ImageTk.PhotoImage(image=testResized)
testLabel = Label(imageFrame, image=testImage)
testLabel.grid(row=1, column=0, padx=10, pady=10)

resultImage = configImages.resizeImage('external/ayang.jpg')
closestResult = ImageTk.PhotoImage(image=resultImage)
resultLabel = Label(imageFrame, image=closestResult)
resultLabel.grid(row=1, column=1, padx=10, pady=10)

# Source file
path = 'None'

srcFrame = LabelFrame(root, width=570, height=20, relief=FLAT, bg='#FFEADF')
srcFrame.grid_propagate(0)
srcFrame.pack(pady=(10, 0))

srcLabel = Label(srcFrame, text=f"Source : {path}", anchor='e', bg='#FFEADF', font=(
    "Arial", 10, "bold"),).grid(
    column=0, row=0, sticky='w', padx=20)

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
    filename = filedialog.askopenfile()
    if (filename):
        testDir.config(text=filename.name)
        testDir.update_idletasks()

        testResized = configImages.resizeImage(filename.name)
        testImage = ImageTk.PhotoImage(image=testResized)
        testLabel.config(image=testImage)
        testLabel.update_idletasks()


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
    start = timer()

    # process

    end = timer()
    elapsedTime = end-start

    timeLabel.config(text=f'Execution time : {elapsedTime:.2f} seconds')
    timeLabel.update_idletasks()


generateButton = Button(descFrame, text='Generate', font=("Montserrat", 12, "bold"), bg='#1F307C', fg='#FFFFFF', width=10, command=generate).grid(
    row=1, column=2, padx=(5, 0), pady=5)

# Live Detect


def detectLive():
    global imgResized

    img = configImages.takePhoto()

    imgResized = cv2.resize(img, (256, 256))
    imgResized = cv2.cvtColor(imgResized, cv2.COLOR_BGR2RGB)

    return imgResized


def detect():
    global testImage
    # img = configImages.takePhoto()
    img = detectLive()

    testDir.config(text="Test Image from Live Camera")
    testDir.update_idletasks()

    # testResized = configImages.convertFrame(img)
    testImage = ImageTk.PhotoImage(image=Image.fromarray(img))
    testLabel.config(image=testImage)
    testLabel.update_idletasks()


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
