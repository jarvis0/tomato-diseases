
# coding: utf-8

# In[5]:


import os
from PIL import Image
import numpy as np


# In[31]:


def normalize(image):
    image = image.astype('float')
    for i in range(3):
        minval = image[...,i].min()
        maxval = image[...,i].max()
        if minval != maxval:
            image[...,i] -= minval
            image[...,i] *= (255.0/(maxval-minval))
    return image


# In[32]:


def apply_normalization(src):
    for img in os.listdir(src):
        image = Image.open(src + img)
        image = np.array(image)
        new_img = Image.fromarray(normalize(image).astype('uint8'),'RGB')
        
        new_img.save(src + img.replace('.jpg','.png'), 'png')


# In[ ]:


folders = ['train', 'val']
classes = sorted(os.listdir("augmented_data/" + folders[0]))

for folder in folders:
    for clss in classes:
        src = "augmented_data/" + folder + '/' + clss + '/'
        apply_normalization(src)

