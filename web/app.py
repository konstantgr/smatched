import io

import streamlit as st
from PIL import ImageEnhance, Image


# from diffusion.image_processor import process_image


def process_image(img_bytes: bytes):
    img = Image.open(io.BytesIO(img_bytes))
    filtered_image = ImageEnhance.Color(img)
    filtered_image.enhance(0)
    return img


def image_form():
    col1, col2 = st.columns(2)
    image_reference, image_source = None, None
    with col1:
        uploaded_file = st.file_uploader("Choose a reference image")
        if uploaded_file is not None:
            image_reference = uploaded_file.getvalue()
            st.image(image_reference)

    with col2:
        uploaded_file = st.file_uploader("Choose a sample image")
        if uploaded_file is not None:
            image_source = uploaded_file.getvalue()
            st.image(image_source)

    generate_button = st.button("Generate", type="primary")
    if generate_button:
        with col2:
            st.image(process_image(image_source))


def run_server():
    st.header("Smatched!")
    image_form()


if __name__ == "__main__":
    run_server()
