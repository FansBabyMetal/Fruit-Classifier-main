import time
import urllib
import numpy as np
from PIL import Image
import streamlit as st
from tensorflow import keras

# Judul Aplikasi
html_temp = """
    <div style="padding-bottom: 20px; padding-top: 20px; padding-left: 5px; padding-right: 5px">
    <center><h1>Fruit Classifier</h1></center>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)

html_temp_list = """
    <div>
    <center><h3>Please upload any Fruit Image from the given list</h3></center>
    <center><h3> [Apple, Banana, Orange, Mango, Pineapple] </h3></center>
    </div>
    """
st.markdown(html_temp_list, unsafe_allow_html=True)

# Inisialisasi variabel gambar agar tidak error
img_to_show = None

opt = st.selectbox(
    "How do you want to upload the image for classification?\n",
    ("Please Select", "Upload image via link", "Upload image from device"),
)

if opt == "Upload image from device":
    file = st.file_uploader("Select", type=["jpg", "png", "jpeg"])
    if file is not None:
        img_to_show = Image.open(file)

elif opt == "Upload image via link":
    img_url = st.text_input("Enter the Image Address")
    if img_url:
        try:
            img_to_show = Image.open(urllib.request.urlopen(img_url))
        except:
            st.error("Please Enter a valid Image Address!")

# Proses Klasifikasi
if img_to_show is not None:
    st.image(img_to_show, width=300, caption="Uploaded Image")

    if st.button("Classify"):
        try:
            # PERBAIKAN: Gunakan Image.LANCZOS atau Image.Resampling.LANCZOS 
            # karena ANTIALIAS sudah dihapus di Pillow versi terbaru
            img_resized = img_to_show.resize((128, 128), Image.LANCZOS)
            
            img_array = np.array(img_resized)
            img_array = img_array.astype("float32") / 255.0

            # Pastikan file model ada di folder Model/model.h5
            model_dir = "Model/model.h5"
            model = keras.models.load_model(model_dir)

            # Labels
            train_labels = {
                "Apple": 0, "Banana": 1, "Mango": 2, "Orange": 3, "Pineapple": 4,
            }
            labels = {v: k for k, v in train_labels.items()}

            # Predicting
            # Menambah dimensi batch (1, 128, 128, 3)
            predictions = model.predict(np.expand_dims(img_array, axis=0))
            acc = np.max(predictions[0]) * 100
            result = labels[np.argmax(predictions[0])]

            # Displaying output
            st.info(f'The uploaded image has been classified as "{result}" with confidence {acc:.2f}%.')

        except Exception as e:
            st.error(f"Error processing image: {e}")