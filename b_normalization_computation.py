
# coding: utf-8

# In[4]:


import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os


# In[5]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = 'augmented_data/'


# In[7]:


data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ])
}

image_dataset = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                    data_transforms[x]) for x in ['train', 'val']}

dataloader = {x: torch.utils.data.DataLoader(image_dataset[x], batch_size=64, shuffle=False, num_workers=4) for x in ['train', 'val']}

pop_mean = []
pop_std0 = []
for phase in ['train', 'val']:
    for data in dataloader[phase]:
        img, labels = data
        #shape (batch_size, 3, height, width
        numpy_image = img.numpy()

        # shape (3,)
        batch_mean = np.mean(numpy_image, axis=(0,2,3))
        batch_std0 = np.std(numpy_image, axis=(0,2,3))

        pop_mean.append(batch_mean)
        pop_std0.append(batch_std0)

# shape (num_iterations, 3) -> (mean across 0th axis) -> shape (3,)
pop_mean = np.array(pop_mean).mean(axis=0)
pop_std0 = np.array(pop_std0).mean(axis=0)

print(pop_mean)
print(pop_std0)

