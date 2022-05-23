import streamlit as st
from datetime import time, datetime
import torch
import matplotlib
import numpy as np
from  PIL import Image


if 'count' not in st.session_state:
	st.session_state.count = 1


uploaded_file = st.file_uploader("Choose a file")

col1, col2 = st.columns(2)


with col1:
  if uploaded_file is not None:
    image=Image.open(uploaded_file)
    #st.write(uploaded_file)
    st.image(image, caption='Sunrise by the mountains')
    st.write('👈  Please upload an image ')

