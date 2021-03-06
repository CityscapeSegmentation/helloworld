import streamlit as st
from datetime import time, datetime
import torch
import matplotlib
import numpy as np
from  PIL import Image
from model import UNet
from utils import trans
from utils import  givin_colors
from utils import  target_names
from utils import  AddTextToMask
from utils import crf
from sklearn.metrics import classification_report
import pandas as pd




if 'count' not in st.session_state:
	st.session_state.count = 1

if 'flag' not in st.session_state:
	#st.session_state.model = UNet(15)
	deep_model=UNet(15)
	deep_model.cpu()
	deep_model.load_state_dict(torch.load('weights/best_cpu2.pt',map_location ='cpu'))
	st.session_state.flag=True
	st.session_state.model=deep_model
else:
	deep_model=st.session_state.model

	

#uploaded_file = st.file_uploader("Choose a file")

col1, col2 = st.columns(2)


# increment = st.button('Next')
# if increment:
#     st.session_state.count += 1

# # A button to decrement the counter
# decrement = st.button('Decrement')
# if decrement:
#     st.session_state.count -= 1


with col1:
   decrement = st.button('Prev')
   if decrement:
       st.session_state.count -= 1
   if st.session_state.count<1:
       st.session_state.count=1

with col2:
   increment = st.button('Next')
   if increment:
       st.session_state.count += 1
   if st.session_state.count>500:
      st.session_state.count=500

target=int(st.session_state.count)
rgb_path='data/val/rgb/'+str(target)+'.png'
mask_path='data/val/mask/'+str(target)+'.png'
  
rgb=Image.open(rgb_path)
image=np.array(rgb)
cpy_image=image.copy()
image = trans(image)

image=image.float()



image=torch.unsqueeze(image, 0).cpu()
deep_model.cpu()

preds=deep_model(image)

values,indecies=torch.max(preds,dim=1)

indecies=indecies.squeeze().cpu().numpy()







mask=Image.open(mask_path)

mask=np.array(mask,dtype=np.uint8)

shape=mask.shape


with col1:
   st.image(rgb, caption=str(target)+'.png')
with col2:


   print('mask shape',mask.shape)
 
   colored_mask=AddTextToMask(mask,target_names)
   st.image(colored_mask, caption=' Mask'+str(target)+'.png')

st.write(indecies.shape)

pred=np.array(indecies,dtype=np.uint8)


#colored_pred=givin_colors[colored_pred]
#colored_pred=colored_pred.reshape((shape[0],shape[1],3))
colored_pred=AddTextToMask(pred,target_names)


colored_img2=givin_colors[pred.reshape(-1)]

colored_img2=colored_img2.reshape((shape[0], shape[1],-1 ))





#st.write(classification_report(mask.reshape(-1), pred.reshape(-1), target_names=target_names))

#print(classification_report(mask.reshape(-1), pred.reshape(-1),target_names=target_names)        )

print('mask unique',np.unique(mask))
print('pred unique',np.unique(pred))

for i in range(15):
   pred[0,i]=i


report_dict=classification_report(mask.reshape(-1), pred.reshape(-1),target_names=target_names, output_dict=True)




df=pd.DataFrame(report_dict)

df1=df.T

st.dataframe(df1)

col3, col4= st.columns(2)

with col3:  
   st.image(colored_pred, caption=' Preds'+str(target)+'.png')





# #pred
# print('pred=',pred.shape)
# crfimage=crf(cpy_image,pred,'after_crf.png')
# if len( crfimage.shape)>2:
#    crfimage=crfimage[:,:,0]
# print('crfimage.shape=',crfimage.shape)
# colored_crfimage=AddTextToMask(crfimage,target_names)





# with col4:  
#    st.image(colored_crfimage, caption=' crf'+str(target)+'.png')

 

# for i in range(15):
#    crfimage[0,i]=i
 
# report_dict=classification_report(mask.reshape(-1), crfimage.reshape(-1),target_names=target_names, output_dict=True)


# df=pd.DataFrame(report_dict)

# df1=df.T

# st.dataframe(df1)
