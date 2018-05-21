
# coding: utf-8

# In[1]:


import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import random

get_ipython().magic('matplotlib inline')


# In[2]:


def show_image(image):
    plt.imshow(np.asarray(image))
    plt.pause(0.001)


# In[3]:


data_dir = '../original_dataset/'
classes = os.listdir(data_dir)
crop_size = 8
crop_list = []

for clss in classes:
    src = data_dir + clss + '/'
    for image in os.listdir(src):
        im = Image.open(src + image)
        black_counter = 0
        for item in im.getdata():
            if item[0] < 15 and item[1] < 10 and item[2] < 15:
                black_counter += 1
        if(black_counter < crop_size * crop_size / 2):
            im = im.crop((0, 0, crop_size, crop_size))
            crop_list.append(im)


# In[ ]:


def background_calculator():    
    im_size = 256
    image = Image.new('RGB', (im_size, im_size))
    for i in range(0,im_size//crop_size):
        for j in range (0,im_size//crop_size):
            random_background = crop_list[random.randint(0, len(crop_list) - 1)]
            image.paste(random_background, (i * crop_size, j * crop_size))
    return image


# In[ ]:


def apply_background(src):
    for img in os.listdir(src):
        background = background_calculator()
        image = Image.open(src + img)
        background.paste(image, (0,0), image)
        background.save(src + img, "PNG")


# In[ ]:


folders = ['val', 'train']
classes = sorted(os.listdir(folders[0]))

for folder in folders:
    for clss in classes:
        src = folder + '/' + clss + '/'
        apply_background(src)

