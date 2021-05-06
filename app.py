import streamlit as st
from PIL import Image, ImageOps
from model import test
import torchvision
import torch

st.write("""
    # Style Transfer Image
""")
st.write("This is a simple style transfer app")
content_file = st.file_uploader(
    "Please upload content image file", type=["jpg", "png"])
style_file = st.file_uploader(
    "Please upload style image file", type=["jpg", "png"])

if style_file is None or content_file is None:
    st.text("You need to upload both style and content image")
else:
    content_image = Image.open(content_file)
    style_image = Image.open(style_file)
    try:
        st.image(content_image)
        st.image(style_image)
        output_tensor = test(
            content_image, style_image).permute(1, 2, 0).numpy()
        output_file = torchvision.transforms.ToPILImage()(output_tensor)
        st.image(output_file)
    except:
        st.text('Unknown error happened')
