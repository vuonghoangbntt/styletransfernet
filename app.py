import streamlit as st
import os
from PIL import Image, ImageOps
from model import test
import torchvision
import torch

st.write("""
    # Style Transfer Image
""")
st.write("This is a simple style transfer app")
content_file = st.file_uploader(
    "Please upload content image file", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader(
    "Please upload style image file", type=["jpg", "jpeg", "png"])
option = st.selectbox(
    'What model do you like?',
    ('Our Model', 'Paper model'))
if style_file is None or content_file is None:
    st.text("You need to upload both style and content image")
# else:
if st.button('Submit'):
    content_image = Image.open(content_file)
    style_image = Image.open(style_file)
    if option == 'Our Model':
        mode = 1
    else:
        mode = 3
    try:
        file_name = test(
            content_image, style_image, mode)
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
