import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import matplotlib

with open('model_faces.h5', 'rb') as f:
    model = load_model('model_faces.h5')

def load_images_faces_test(ima):
    ret = []
    pieces = []
    img_array = np.array(ima)
    for i in range(6):
        for j in range(6):
            pieces.append(img_array[i*50:(i+1)*50, j*50:(j+1)*50])
    ret.append(pieces)
    return np.array(ret)

# a function to rearrange the puzzle pieces to their right positions
def rearrange(df, image, idx):
    # img_name = df.loc[idx, "image"]
    im = Image.fromarray(image)
    new_im = np.zeros_like(image)
    cut = image.shape[0] // 6
    for i in range(6):
        for j in range(6):
            r, c = int(df.loc[idx, str(i)+str(j)][0]), int(df.loc[idx, str(i)+str(j)][1])
            new_im[r*cut:(r+1)*cut, c*cut:(c+1)*cut] = np.array(im.crop((j*cut, i*cut, (j+1)*cut, (i+1)*cut)))
    return new_im

def predict(filename, image):

    imageinp = load_images_faces_test(image)

    # Make the prediction
    pred = model.predict(imageinp)
    pred = np.argmax(pred, axis=-1)

    pred_list = []
    for i in range(pred.shape[0]):
        t = []
        for j in range(pred[i].shape[0]):
            t.append(str(pred[i][j]//6) + str(pred[i][j]%6))
        pred_list.append(t)

    label_df = pd.DataFrame(pred_list, columns=['00','01','02','03','04','05','10','11','12','13','14','15','20','21','22','23','24','25','30','31',
                    '32','33','34','35','40','41','42','43','44','45','50','51','52','53','54','55'])

    # combine filename and label dataframe
    faces_df = pd.concat([pd.DataFrame([filename], columns=['image']), label_df], axis=1)
    
    print(faces_df)
    # rearrange puzzle pieces to their right positions
    new_im = rearrange(faces_df, image, idx=0)

    
    # Return the new image
    new_im = np.array(new_im, dtype = np.uint8)
    return new_im

st.title('Jigsaw Puzzle Solver: ')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    # Read the image
    img_bytes = uploaded_file.read()
    buffer_size = len(img_bytes)
    # Ensure buffer size is a multiple of element size
    buffer_size -= buffer_size % np.dtype(np.uint8).itemsize
    image = cv2.imdecode(np.frombuffer(img_bytes[:buffer_size], np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    filename = uploaded_file.name

    # Make the prediction
    pred = predict(filename, image)

    # fig, ax = plt.subplots(1, 2)
    # ax[0].imshow(image)
    # ax[0].set_title("puzzle")
    # ax[1].imshow(pred)
    # ax[1].set_title("predicted")
    
    # st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Input Image", clamp=False, channels="RGB", output_format="auto")

    with col2:
        st.image(pred, caption="Predicted Image", clamp=False, channels="RGB", output_format="auto")
