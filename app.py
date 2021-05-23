import streamlit as st
import os
from PIL import Image, ImageOps
from model import test
import torchvision
import torch
st.sidebar.title("""
    # Style Transfer Image
""")
st.sidebar.subheader("This is a simple style transfer app")
st.sidebar.subheader('Alpha')
alpha = st.sidebar.slider('A number between 0.5-1',
                          min_value=0.5, max_value=1.0, step=0.01, value=1.0)
content_file = st.sidebar.file_uploader(
    "Please upload content image file", type=["jpg", "jpeg", "png"])
style_file = st.sidebar.file_uploader(
    "Please upload style image file", type=["jpg", "jpeg", "png"])
option = st.sidebar.selectbox(
    'What model do you like?',
    ('Our Model', 'Paper model'))
if style_file is None or content_file is None:
    st.sidebar.text("You need to upload both style and content image")
# else:
if st.sidebar.button('Submit'):
    content_image = Image.open(content_file)
    style_image = Image.open(style_file)
    if option == 'Our Model':
        mode = 1
    else:
        mode = 3
    try:
        file_name = test(
            content_image, style_image, mode, alpha)
        file_z = Image.open(file_name)
        list_image = [content_image, style_image, file_z]
        st.image(list_image, width=400)
    except:
        st.text('Unknown error happened')
    if file_name is not None:
        try:
            os.remove(file_name)
        except:
            st.text('can not delete file')
