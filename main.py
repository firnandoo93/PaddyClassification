import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# Load the trained model
model_path = "D:\Machine Learning\PaddyClassification\InceptionV3.h5"
loaded_model = tf.keras.models.load_model(model_path)

# Define the class labels
class_labels = ["Bacterialblight", "Blast", "Brownspot", "Tungro"]

st.set_page_config(
    page_title="Classify",
    page_icon=":herb:",
    layout="wide"
)

# Container
with st.container():
    cl1, cl2, cl3 = st.columns((1,1,1))
    with cl2:
        st.title('KLASIFIKASI PENYAKIT PADI')
    st.write("---")

with st.container():
    cl1, cl2, cl3 = st.columns((1,2,1))
    with cl2:
        # Upload image through Streamlit
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "JPG", "JPEG"], accept_multiple_files=False)


with st.container():
    cl1, cl2, cl3 = st.columns((2,2,2))
    with cl2:
        if uploaded_file is not None:
            # Preprocess the image for prediction
            imgPrev = Image.open(uploaded_file)
            # Display the uploaded image with a specified pixel width
            st.image(imgPrev, use_column_width=False, width=500, output_format="JPEG")

            img = Image.open(uploaded_file)
            img = img.resize((299, 299))
            img_array = np.array(img)  # Convert the image to a NumPy array
            img_array = img_array / 255.0  # Normalize pixel values


            # Add a button to trigger classification below the image
            classify_button = st.button("Classify")
            st.markdown("""
                <style>
                    div.stButton button {
                        background: #FDF0D1;
                        width: 500px; 
                        height: 50px; 
                        font-size: 18px; 
                    }
                </style>
            """, unsafe_allow_html=True)

            if classify_button:
                # Make prediction
                prediction = loaded_model.predict(np.expand_dims(img_array, axis=0))
                predicted_class = np.argmax(prediction)

                # Display the prediction result
                st.subheader("Prediction:")
                st.write(f"Gambar daun padi diatas diklasifikasikan sebagai {class_labels[predicted_class]}")
                