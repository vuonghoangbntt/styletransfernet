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
    ('Hiep Model', 'Hoang Model'))
if style_file is None or content_file is None:
    st.text("You need to upload both style and content image")
# else:
if st.button('Submit'):
    content_image = Image.open(content_file)
    style_image = Image.open(style_file)
    if option == 'Hiep Model':
        mode = 2
    else:
        mode = 1
    try:
        #st.image(content_image, width=200)
        #st.image(style_image, width=200)
        file_name = test(
            content_image, style_image, mode)
        file_z = Image.open(file_name)
        #output_file = torchvision.transforms.ToPILImage()(output_tensor)
        list_image = [content_image, style_image, file_z]
        st.image(list_image, width=400)
        os.remove(file_name)
    except:
        st.text('Unknown error happened')
