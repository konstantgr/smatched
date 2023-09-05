from typing import Optional

import streamlit as st
from PIL import Image
from streamlit.runtime.uploaded_file_manager import UploadedFile

from web import MAIN_FOLDER


def display_images(source_image: Optional[type(Image)], reference_image: Optional[type(Image)]) -> None:
    cols = st.columns(2)
    subheaders_mapping = {
        0: "Source image",
        1: "Reference image"
    }

    for i, image in enumerate([source_image, reference_image]):
        if image is not None:
            cols[i].subheader(subheaders_mapping[i])
            cols[i].image(image, use_column_width=True)


def get_default_reference_image() -> Image:
    img = Image.open(MAIN_FOLDER / "images/reference_image.png")
    return img


def run_server():
    st.title("Smatched!")

    st.header("Image Upload and Text Input")

    images_source, image_reference = None, None
    col1, col2, col3 = st.columns(3)

    with col1:
        text_input = st.text_input("Enter some text:")

    with col2:
        images_source = st.file_uploader(
            "Upload source image",
            type=["jpg", "png", "jpeg"],
        )

    with col3:
        image_reference = st.file_uploader(
            "Upload reference image",
            type=["jpg", "png", "jpeg"]
        )

    st.divider()

    if image_reference is None:
        image_reference = get_default_reference_image()

    if images_source or image_reference:
        uploaded_images = []

        if images_source:
            uploaded_images.append(Image.open(images_source))

        if isinstance(image_reference, UploadedFile):
            uploaded_images.append(Image.open(image_reference))

        with st.info("Uploaded Images:"):
            display_images(images_source, image_reference)
        st.divider()


if __name__ == "__main__":
    run_server()
