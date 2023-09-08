import streamlit as st
from PIL import Image
from web import MAIN_FOLDER
from web.run_processor import run_imagine_processor
from streamlit_image_comparison import image_comparison


def get_default_reference_image() -> Image:
    img = Image.open(MAIN_FOLDER / "images/reference_image.png")
    return img


def run_server():
    st.set_page_config(layout="wide")

    st.title("SmatcheD!")
    st.subheader("Image Upload and Text Input")

    image_source, image_reference, generated_image = None, None, None
    col1, col2, col3 = st.columns(3, gap='large')

    with col1:
        text_input = st.text_input("Enter some text:", value='nude makeup')

    with col2:
        image_source = st.file_uploader(
            "Upload source image",
            type=["jpg", "png", "jpeg"],
        )

    with col3:
        image_reference = st.file_uploader(
            "Upload reference image",
            type=["jpg", "png", "jpeg"]
        )
    generate_button = st.button("Generate makeup")
    st.divider()

    if image_source is None:
        image_source = get_default_reference_image()

    if image_source or image_reference:
        cols = st.columns(3, gap='large')
        subheaders_mapping = {
            0: "Generated image",
            1: "Source image",
            2: "Reference image"
        }

        processor = None
        if generate_button:
            with cols[0]:
                st.subheader(subheaders_mapping[0])
                with st.spinner('Wait for it...'):
                    generated_image, cropped_image, processor = run_imagine_processor(processor, image_source,
                                                                                      text_input)

                    if generated_image is not None:
                        image_comparison(cropped_image, generated_image, width=256)
                    else:
                        st.empty()

        for i, image in enumerate([image_source, image_reference], 1):
            cols[i].subheader(subheaders_mapping[i])
            if image is not None:
                cols[i].image(image, width=256)
            else:
                cols[i].text('Not loaded yet')


if __name__ == "__main__":
    run_server()
