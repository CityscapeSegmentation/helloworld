


from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader

import numpy as np


    
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
