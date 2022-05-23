import streamlit as st
from datetime import time, datetime
import torch
import matplotlib
import numpy as np
from  PIL import Image
from model import UNet
from utils import trans





if 'count' not in st.session_state:
	st.session_state.count = 1

if 'flag' not in st.session_state:
	#st.session_state.model = UNet(15)
	deep_model=UNet(15)
	deep_model.cpu()
	deep_model.load_state_dict(torch.load('weights/best_cpu.pt',map_location ='cpu'))
	st.session_state.flag=True
	#st.session_state.model=model

	

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
image=np.array(rgb)
image = trans(image)

image=torch.unsqueeze(image, 0)

preds=deep_model(image)

values,indecies=torch.max(preds,dim=1)

indecies=indecies.squeeze().cpu().nump()


st.write(image.shape)




mask=Image.open(mask_path)

mask_brighter=np.array(mask)


with col1:
   st.image(rgb, caption=str(target)+'.png')
with col2:
   st.image(15*mask_brighter, caption=' Mask'+str(target)+'.png')

st.image(15*indecies, caption=' Preds'+str(target)+'.png')




