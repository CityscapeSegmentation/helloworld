


from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader
import streamlit as st

import numpy as np
import cv2
from PIL import ImageDraw,ImageFont,Image


import numpy as np
import pydensecrf.densecrf as dcrf
import matplotlib.pyplot as plt

from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

from skimage.color import gray2rgb
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
from osgeo import gdal
#####################################################
"""
Function which returns the labelled image after applying CRF

"""

#Original_image = Image which has to labelled
#Annotated image = Which has been labelled by some technique( FCN in this case)
#Output_path = Name of the final output image after applying CRF
#Use_2d = boolean variable 
#if use_2d = True specialised 2D fucntions will be applied
#else Generic functions will be applied

def crf(original_image, predicted_mask,Output_path, use_2d = True):
    
    # Converting annotated image to RGB if it is Gray scale
    if(len(predicted_mask.shape)<3):
        predicted_mask = gray2rgb(predicted_mask).astype(np.uint32)
    
    #cv2.imwrite("testing2.png",predicted_mask)
    predicted_mask = predicted_mask.astype(np.uint32)
    #Converting the annotations RGB color to single 32 bit integer
    annotated_label = predicted_mask[:,:,0].astype(np.uint32) + (predicted_mask[:,:,1]<<8).astype(np.uint32) + (predicted_mask[:,:,2]<<16).astype(np.uint32)
    
    # Convert the 32bit integer color to 0,1, 2, ... labels.
    colors, labels = np.unique(annotated_label, return_inverse=True)
    
    #Creating a mapping back to 32 bit colors
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16
    
    #Gives no of class labels in the annotated image
    n_labels = len(set(labels.flat)) 
    
    print("No of labels in the Image are ")
    print(n_labels)
    
    
    #Setting up the CRF model
    if use_2d :
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.90, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                          normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
                           compat=10,
                           kernel=dcrf.DIAG_KERNEL,
                           normalization=dcrf.NORMALIZE_SYMMETRIC)
        
    #Run Inference for 5 steps 
    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    # Note that there is no "unknown" here anymore, no matter what we had at first.
    MAP = colorize[MAP,:]
    cv2.imwrite(Output_path,MAP.reshape(original_image.shape))
    return MAP.reshape(original_image.shape)

#####################################################
    
trans = transforms.Compose([
    transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
])    

target_names=['avoid','ground','road','sidewalk','parking','building','fence or guard rail',
             'traffic sign or pole','vegetation','terrain','sky','person','vehicle','bike','wall']


class CityDataset(Dataset):
    def __init__(self, data_path,  transform=None):
        self.input_images=os.listdir(data_path+'/rgb')        
        self.transform = transform
        self.data_path=data_path
    
    def __len__(self):
        return len(self.input_images)
    
    def __getitem__(self, idx):     
        
        data_path=self.data_path
        
        image = np.array(Image.open( data_path+'/rgb/'+ self.input_images[idx])  )
        mask = np.array(Image.open( data_path+'/mask/'+ self.input_images[idx]))
        
        if self.transform:
            image = self.transform(image)
            mask=torch.from_numpy(mask).long()
            #mask=mask.long()
        
        return [image, mask]
    
#val_set = CityDataset(data_path='data/val', transform = trans)
#DataLoader(val_set, batch_size=1, shuffle=True, num_workers=0)

#image = np.array(Image.open( data_path+'/rgb/'+ self.input_images[idx])  )

tmp=[
[0,0,0],
[0,0,0],
[0,0,0],
[0,0,0],
[0,0,0],
[111,74,0],
[81,0,81],
[128,64,128],
[244,35,232],
[250,170,160],
[230,150,140],
[70,70,70],
[102,102,156],
[190,153,153],
[180,165,180],
[150,100,100],
[150,120,90],
[153,153,153],
[153,153,153],
[250,170,30],
[220,220,0],
[107,142,35],
[152,251,152],
[70,130,180],
[220,20,60],
[255,0,0],
[0,0,142],
[0,0,70],
[0,60,100],
[0,0,90],
[0,0,110],
[0,80,100],
[0,0,230],
[119,11,32],
[0,0,142]]

###remove repeated colors
givin_colors=[]
for c in tmp:
    if not c in givin_colors:
        givin_colors.append(c)
        #print(c)

givin_colors=np.array(givin_colors)
#print(givin_colors.shape)


def PlotText(mask_,target_names_list):
    unq=np.unique(mask_).tolist()[1:]
    print(np.unique(mask_).tolist())
    text_pos={}
    #print('***********')
    for f in unq:
        thresh=0*mask_
        thresh[np.where(mask_==f)]=255       
        
        # You need to choose 4 or 8 for connectivity type
        connectivity = 8
        # Perform the operation to get information about regoins!!!
        try:
           
            shape=thresh.shape
            if len(shape)>2:               
                thresh=thresh[:,:,0]
            #cv2.imshow('thresh',thresh)
            #cv2.waitKey(0)
            output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
        except Exception as e:
            st.write(e)
        # Get the results
        # The first cell is the number of labels
        num_labels = output[0]
        # The second cell is the label matrix

        labels = output[1]
        # The third cell is the stat matrix

        #print(np.max(labels))
        
        
        radius = 20
  
        # Blue color in BGR
        color = (255, 0, 0)

        # Line thickness of 2 px
        thickness = 2



        stats = output[2]
        
        #print(target_names_list[f])
        #print(stats)
        
        
        # The fourth cell is the centroid matrix
        centroids = output[3]

        

        im=cv2.merge((thresh,thresh,thresh))
        #print(im.shape)
        Flag=False
        current_class=target_names_list[f]
        
        text_pos[current_class]=[]
        
        
        for i in range(1,stats.shape[0]):
            if stats[i][4]>500: #number of pixels bigger than 500 pixels
                
                #im=cv2.rectangle(im, (stats[i][0], stats[i][1]), (stats[i][0]+stats[i][2], stats[i][1]+stats[i][3]), (0, 255, 0), 1)
                Flag=True
                #print(centroids[i])                                
                x,y=centroids[i]
                x,y=int(x),int(y)
                
                text_pos[current_class].append((x,y))
                #print('***********************')
                
                
  
                # Using cv2.circle() method
                # Draw a circle with blue line borders of thickness of 2 px
                #im = cv2.circle(im,(x,y) , radius, color, thickness)
        

                
        if Flag==True:
            #print(f,target_names_list[f])
            #plt.imshow(im)
            #plt.show()
            Flag=False
            
    return text_pos


def AddTextToMask(mask,target_names):
    
    shape=mask.shape
    
    text_pos=PlotText(mask,target_names)
    colored_img=givin_colors[mask.reshape(-1)]
    
    colored_img=colored_img.reshape((shape[0],shape[1],3))  
    
    colored_img=np.array(colored_img,dtype=np.uint8)
    
    pil_im = Image.fromarray(colored_img)  

    for k in text_pos:
         if len(text_pos[k])>0:
                #print(k,len(text_pos[k]))          

                text=k

                draw = ImageDraw.Draw(pil_im)  
                # use a truetype font  
                font = ImageFont.truetype("Aaron-BoldItalic.ttf", 10)  

                coords=text_pos[k]
                # Draw the text  
                for c in coords:
                    x,y=c
                    draw.text((x,y), text, font=font,fill=(255,255,255,0))  

    new_mask=np.array(pil_im)
    return new_mask
    
 