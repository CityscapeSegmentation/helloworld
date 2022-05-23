import streamlit as st
from datetime import time, datetime
import torch
import matplotlib
import numpy as np
from  PIL import Image
from model import UNet


if 'count' not in st.session_state:
	st.session_state.count = 1

if 'model' not in st.session_state:
	st.session_state.model = UNet(15)
	model=st.session_state.model
	model.load_state_dict(torch.load('weights/best.pt'))
	

#uploaded_file = st.file_uploader("Choose a file")

col1, col2 = st.columns(2)


# increment = st.button('Increment')
# if increment:
#     st.session_state.count += 1

# # A button to decrement the counter
# decrement = st.button('Decrement')
# if decrement:
#     st.session_state.count -= 1


with col1:
   decrement = st.button('Decrement')
   if decrement:
       st.session_state.count -= 1
   if st.session_state.count<1:
       st.session_state.count=1

with col2:
   increment = st.button('Increment')
   if increment:
       st.session_state.count += 1
   if st.session_state.count>500:
      st.session_state.count=500

target=int(st.session_state.count)
rgb_path='data/val/rgb/'+str(target)+'.png'
mask_path='data/val/mask/'+str(target)+'.png'
  
rgb=Image.open(rgb_path)
mask=Image.open(mask_path)

mask=15*np.array(mask)


with col1:
   st.image(rgb, caption=str(target)+'.png')
with col2:
   st.image(mask, caption=' Mask'+str(target)+'.png')

#st.write('👈  Please upload an image ')

