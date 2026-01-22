import streamlit as st
from blip_model import BlipCaptioner
import tempfile

st.set_page_config(page_title="BLIP Captioner", layout="centered")
st.title("BLIP Image Captioner")

captioner = BlipCaptioner()

uploaded_file = st.file_uploader(
    "Choose an image...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    temp_file.close()


    st.image(temp_file.name, caption="Uploaded image", width=500)


    with st.spinner("Generating caption..."):
        text = captioner.caption(temp_file.name)

    st.subheader("Image caption")
    st.success(text)
