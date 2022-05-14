from calendar import c
import os
from this import d
import cv2
import time
import pickle
import imutils
from recog import *
import streamlit as st
import face_recognition
from imutils import paths

# wide mode
st.set_page_config(layout="wide")
st.title('Face Recognition using Artificial Intelligence')

with open('readme.md') as f:
    with st.expander("view Instruction to install"):
        st.markdown(f.read())   
options = ['Open Camera','Add Person','Delete Person','Train Model']
col1,col2 = st.columns([1,1])  
col1.header('Output')
col2.header('Options')
choice = col2.radio('Choose an option',options)  
if choice == 'Open Camera':
    if col1.button('Open Camera 🎥'):
        col1.markdown('''
        ### Camera opens in a new window
        - to close the camera, click on the window and press `q`
        - dont press the cross button on the top right corner
        ''')
        with st.spinner('Opening Camera, please wait loading faces...'):
            time.sleep(2)
            start_camera(load_face_encodings())

if choice == 'Add Person':
    col1.markdown('''
    ### Add a new person
    - Camera opens in a new window
    - dont press the cross button on the top right corner
    - press `'s'` to save the image
    - press `'q'` to quit
    ''')
    name = col1.text_input('Enter Name')
    designation = col1.text_input('Enter Designation')
    if col1.button('Add Person 🧑‍🦰') and name and designation:
        with st.spinner('Opening Camera'):
            save_ur_images(name,designation)

if choice == 'Delete Person':
    content = os.walk('images')
    for root, dirs, files in content:
        for dir in dirs:
            if col1.button(f'Delete {dir} ❌'):
                if delete_dir(dir):
                    st.success(f'{dir} deleted, press `R` to refresh')
                else:
                    st.error("person could not be deleted,check admin")

if choice == 'Train Model':
    if col1.button('Train Model 🧠'):
        with st.spinner('Training Model'):
            save_face_encodings(st,stlit=True)
            st.success('Model Trained')


    