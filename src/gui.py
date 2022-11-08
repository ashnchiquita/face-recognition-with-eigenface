from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog

# initialize root and title
root = Tk()
root.title('Face Recognition with EigenFace')
root.iconbitmap('external/logo.ico')

# Header
header = Label(text='Face Recognition with EigenFace').grid(
    row=0, columnspan=4)

# Image Description
desc1 = Label(text='Test Image').grid(row=1, column=0, columnspan=2)
desc2 = Label(text='Closest Result').grid(row=1, column=2, columnspan=2)

# Image Container
testImage = ImageTk.PhotoImage(Image.open('src/Mean.jpg'))
testLabel = Label(image=testImage)
testLabel.grid(row=2, column=0, columnspan=2)

closestResult = ImageTk.PhotoImage(Image.open('src/Mean.jpg'))
resultLabel = Label(image=closestResult)
resultLabel.grid(row=2, column=2, columnspan=2)

# Choose Test Image

filename = 'No Files Chosen'


def chooseTest():
    global filename
    filename = filedialog.askopenfile()
    testDir.config(text=filename.name)
    testDir.update_idletasks()


testButton = Button(root, text='Choose Test Image',
                    command=chooseTest).grid(row=3, column=0)
testDir = Label(root, text=filename, anchor='e')
testDir.grid(row=3, column=1, columnspan=2)

# Choose Dataset

foldername = 'No Folders Chosen'


def chooseDataset():
    global foldername
    foldername = filedialog.askdirectory()
    dataDir.config(text=foldername)
    dataDir.update_idletasks()


dataButton = Button(root, text='Choose Dataset',
                    command=chooseDataset).grid(row=4, column=0)
dataDir = Label(root, text=foldername, anchor='e')
dataDir.grid(row=4, column=1, columnspan=2, sticky='w')

# Execute Recognize
generateButton = Button(root, text='Generate').grid(row=3, rowspan=2, column=3)

# Execution Time
time = 20
timeLabel = Label(root, text=f'Execution time : {time}').grid(
    row=5, columnspan=4)

Label().grid(padx=20, pady=20, row=7, columnspan=4)
root.mainloop()
