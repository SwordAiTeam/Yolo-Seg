import streamlit as st
from ultralytics import YOLO
from PIL import Image

# -------------------------
# Load model once
# -------------------------
@st.cache_resource
def load_model():
    # assumes best.pt is in the same folder as app.py
    model = YOLO("best.pt")
    return model

model_pt = load_model()

# -------------------------
# Streamlit UI
# -------------------------
st.title("YOLOv8 Segmentation Inference App")
st.write("Upload one or more images to run segmentation with your YOLOv8 model.")

uploaded_files = st.file_uploader(
    "Upload image(s)", 
    type=["png", "jpg", "jpeg"], 
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Open image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_column_width=True)

        # Run inference
        results_pt = model_pt.predict(source=image, conf=0.5, save=False)
        
        # Get result with plotted masks
        im_array_pt = results_pt[0].plot()
        im_pt = Image.fromarray(im_array_pt[..., ::-1])  # BGR -> RGB

        # Show result
        st.image(im_pt, caption=f"Result: {uploaded_file.name}", use_column_width=True)
