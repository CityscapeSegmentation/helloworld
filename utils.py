


from torchvision import transforms, datasets, models
from torch.utils.data import Dataset, DataLoader




    
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
