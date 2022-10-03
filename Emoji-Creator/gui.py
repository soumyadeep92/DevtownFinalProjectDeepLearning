import tkinter as tk
from tkinter import *
import cv2
from PIL import Image, ImageTk
import os
import numpy as np
import sys
import numpy as np
import time
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D
from keras.optimizers import Adam
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

emotion_model = Sequential()

emotion_model.add(Conv2D(64, padding='same', kernel_size=(3, 3), input_shape=(48,48,1)))
emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(5, 5), padding='same'))
emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(512, kernel_size=(3, 3), padding='same'))
emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(256))
emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))
emotion_model.add(Dropout(0.25))
emotion_model.add(Dense(512))
emotion_model.add(BatchNormalization())
emotion_model.add(Activation('relu'))
emotion_model.add(Dropout(0.25))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('C:/Users/Dipak/Desktop/Devtown-Final-SubmissionProject-2022/Emojify/emoji-creator-project-code/emotion_model.h5')

cv2.ocl.setUseOpenCL(False)

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


emoji_dist={0:"C://Users//Dipak//Desktop//Devtown-Final-SubmissionProject-2022//Emojify//Emoji-Creator//emojis//disgusted.png",2:"C://Users//Dipak//Desktop//Devtown-Final-SubmissionProject-2022//Emojify//Emoji-Creator//emojis//fearful.png",3:"C://Users//Dipak//Desktop//Devtown-Final-SubmissionProject-2022//Emojify//Emoji-Creator//emojis//happy.png",4:"C://Users//Dipak//Desktop//Devtown-Final-SubmissionProject-2022//Emojify//Emoji-Creator//emojis//neutral.png",5:"C://Users//Dipak//Desktop//Devtown-Final-SubmissionProject-2022//Emojify//Emoji-Creator//emojis//sad.png",6:"C://Users//Dipak//Desktop//Devtown-Final-SubmissionProject-2022//Emojify//Emoji-Creator//emojis//surprised.png"}

global last_frame1                                    
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text=[0]
def show_vid():   
    
    cap1 = cv2.VideoCapture(0)                                 
    if not cap1.isOpened():                             
        print("cant open the camera1")
    flag1, frame1 = cap1.read()
    frame1 = cv2.resize(frame1,(600,500))
    
    bounding_box = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    num_faces = bounding_box.detectMultiScale(gray_frame,scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame1, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame1, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        show_text[0]=maxindex
    cv2.imshow('frame1',frame1)
    if flag1 is None:
        print ("Major error!")
    elif flag1:
        global last_frame1
        last_frame1 = frame1.copy()
        pic = cv2.cvtColor(last_frame1, cv2.COLOR_BGR2RGB)     
        img = Image.fromarray(pic)
        imgtk = ImageTk.PhotoImage(image=img)
        lmain.imgtk = imgtk
        lmain.configure(image=imgtk)
        lmain.after(10, show_vid) 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        raise KeyboardInterrupt('Press Quit button in subscreen to Exit')

    

def show_vid2():
    frame2=cv2.imread(emoji_dist[show_text[0]])
    pic2=cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
    img2=Image.fromarray(frame2)
    imgtk2=ImageTk.PhotoImage(image=img2)
    lmain2.imgtk2=imgtk2
    lmain3.configure(text=emotion_dict[show_text[0]],font=('arial',45,'bold'))
    
    lmain2.configure(image=imgtk2)
    lmain2.after(10, show_vid2)

if __name__ == '__main__':
    root=tk.Tk()   
    img = ImageTk.PhotoImage(Image.open("logo.png"))
    
    heading2=Label(root,text="Emojify",pady=20, font=('arial',45,'bold'),bg='black',fg='blue')                                 
    heading2.pack()
    heading2.place(x=700,y=5)
    heading = Label(root,image=img,bg='black')
    heading.pack() 
    heading.place(x=700,y=95)
    
    lmain = tk.Label(master=root,padx=50,bd=10,height=600,width=500)
    lmain2 = tk.Label(master=root,bd=10)

    lmain3=tk.Label(master=root,bd=5,fg='white',bg='black')
    lmain.pack(side=LEFT)
    lmain.place(x=80,y=50)
    lmain3.pack()
    lmain3.place(x=940,y=250)
    lmain2.pack(side=RIGHT)
    lmain2.place(x=900,y=350)
    

          
    root.geometry("1400x900") 
    root['bg']='black'
    exitbutton = Button(root, text='Quit',fg="red",command=root.destroy,font=('arial',25,'bold')).pack(side = BOTTOM)
    show_vid()
    show_vid2()
    root.mainloop()