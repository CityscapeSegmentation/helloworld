import streamlit as st
from datetime import time, datetime
import torch
import matplotlib
import numpy as np
from  PIL import Image



uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
  image=Image.open(uploaded_file)
  st.write(uploaded_file)
  st.image(image, caption='Sunrise by the mountains')

