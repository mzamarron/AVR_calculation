import numpy as np
import pandas as pd
import os
import cv2
import streamlit as st
import streamlit.components.v1 as components
import urllib
from PIL import Image
from avr import *

st.set_page_config(layout="wide", page_title='AVR calculation webapp')
st.image('Banner.PNG', use_column_width=True)
st.markdown("<h2 style='text-align: center; color: black;'>Automated Calculation of AVR in Fundus Images</h2>",
            unsafe_allow_html=True)

st.markdown("""
<style>
.small-font {
    font-size:13px !important;
}
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.sidebar.header('Options')
    with st.expander('How to use this tool'):
        st.markdown('---')
        st.write("1. Upload a Fundus image.")
        st.write("2. Upload its Optic Disc label.")
        st.write("3. Upload its A/V Classification.")
        st.write("4. Optionally display all images.")
        st.write("5. Click on [Calculate].")
        st.markdown('---')
        st.markdown('_* Images must be in png format._')
    display_im = st.sidebar.radio('Display uploaded images', ['Yes', 'No'])
    with st.expander('About'):
        st.markdown('---')
        st.write()
        st.markdown('<p class="small-font">This tool was created for the thesis:</p>', unsafe_allow_html=True)
        st.markdown('<p class="small-font"><i>Early Hypertensive Retinopathy detection by leveraging salient regions '
                    'using deep learning </i>', unsafe_allow_html=True)
        st.markdown('<p class="small-font"> by Eng. Monica Zamarron-Perez.</p>', unsafe_allow_html=True)

l0 = st.empty()
_, col1, col2, col3, _ = st.columns([5, 3, 3, 3, 5])
l1 = st.empty()
_, image_location, _ = st.columns(3)
_, _, avr_location, _, _ = st.columns([5, 3, 3, 3, 5])

# Using the "with" syntax
with st.form(key='upload_files_form'):
    st.subheader("Upload images")
    eye_fundus_image = st.file_uploader("Fundus Image", type=['png'])
    od_labels_image = st.file_uploader("Optic Disc Label", type=['png'])
    av_labels_image = st.file_uploader("Arteries/Veins Classification", type=['png'])
    submit_button = st.form_submit_button(label='Calculate')
    if eye_fundus_image is not None and od_labels_image is not None and av_labels_image is not None:
        file = eye_fundus_image.name[:-4]
        print('Processing {}'.format(file))
        df_results = pd.DataFrame()
        # read images with Pillow the convert to numpy arrays BGR
        image = Image.open(eye_fundus_image)
        img_original = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        res_image = img_original.copy()
        for radius in RADIUS:
            # The function cv2.imread() is used to read an image. (BGR)
            image = Image.open(od_labels_image)
            od_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            image = Image.open(av_labels_image)
            av_img= cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            result_image, _df = calculate_avr(res_image, od_img, av_img, file, radius)
            df_results = pd.concat([df_results, _df])
        df_avr = df_results.groupby(['file'], as_index=False).apply(lambda x: avr_calculation(x, radius=RADIUS))
        df_avr.columns = ['file', 'avr']
        av_ratio = df_avr.loc[0, 'avr']
        image_location.image(cv2.cvtColor(np.array(result_image),  cv2.COLOR_BGR2RGB), use_column_width=True)
        avr_location.subheader('AVR = {:.4f}'.format(av_ratio))
        if display_im == 'Yes':
            l0.image('line.png', use_column_width=True)
            col1.image(Image.open(eye_fundus_image), width=200, caption='Fundus image')
            col2.image(Image.open(od_labels_image), width=200, caption='Optic Disc')
            col3.image(Image.open(av_labels_image), width=200, caption='A/V Classification')
            l1.image('line.png', use_column_width=True)
        print('Done {:.4f}'.format(av_ratio))

