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


increment = st.button('Increment')
if increment:
    st.session_state.count += 1

# # A button to decrement the counter
# decrement = st.button('Decrement')
# if decrement:
#     st.session_state.count -= 1


with col1:
   decrement = st.button('Decrement')
   if decrement:
       st.session_state.count -= 1
   if st.session_state.count<1:
       target=st.session_state.count=1

with col2:
   increment = st.button('Increment')
   if increment:
       st.session_state.count += 1
   if st.session_state.count>500:
      target=st.session_state.count=500
	
rgb_path='data/val/rgb/'+str(target)+'jpg'
mask_path='data/val/mask/'+str(target)+'jpg'
  
rgb=Image.open(rgb_path)
mask=Image.open(mask_path)

#st.write(uploaded_file)
st.image(image, caption='Sunrise by the mountains')
st.write('👈  Please upload an image ')

