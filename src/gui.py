from tkinter import *
import tkinter as tk
from PIL import ImageTk, Image
from tkinter import filedialog

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
testImage = ImageTk.PhotoImage(Image.open('src/Mean.jpg'))
testLabel = Label(imageFrame, image=testImage)
testLabel.grid(row=1, column=0, padx=10, pady=10)

closestResult = ImageTk.PhotoImage(Image.open('src/Mean.jpg'))
resultLabel = Label(imageFrame, image=closestResult)
resultLabel.grid(row=1, column=1, padx=10, pady=10)

# Choose Test Image

descFrame = LabelFrame(root, width=580, height=110, relief=FLAT, bg='#FFEADF')
descFrame.grid_propagate(0)
descFrame.pack(pady=20)

descFrame.columnconfigure(0, weight=4)
descFrame.columnconfigure(1, weight=16)
descFrame.columnconfigure(2, weight=4)

filename = 'No Files Chosen'


def chooseTest():
    global filename
    filename = filedialog.askopenfile()
    if (filename):
        testDir.config(text=filename.name)
        testDir.update_idletasks()


testButton = Button(descFrame, text='Choose Test Image',
                    command=chooseTest, width=16, height=2).grid(row=1, column=0, pady=(0, 10))
testDir = Label(descFrame, text=filename, font=(
    "Arial", 10, "bold"), bg='#FFEADF', anchor='e')
testDir.grid(row=1, column=1, sticky='w', pady=(0, 10), padx=5)

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
dataDir.grid(row=2, column=1, sticky='w', pady=(10, 0), padx=5)

# Execute Recognize
generateButton = Button(descFrame, text='Generate', font=("Montserrat", 15, "bold"), bg='#1F307C', fg='#FFFFFF').grid(
    row=1, rowspan=2, column=2, padx=5, pady=5)

# Execution Time
exeFrame = LabelFrame(root)
exeFrame.pack()
time = 20
timeLabel = Label(exeFrame, text=f'Execution time : {time}').grid(
    row=5, columnspan=4)

root.mainloop()
