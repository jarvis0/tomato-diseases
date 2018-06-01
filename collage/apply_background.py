
# coding: utf-8

# In[5]:


from PIL import Image
import os
import random
import shutil
from random import shuffle


# In[8]:


def shuffle_backgrounds():
    backgrounds = sorted(os.listdir('./' + background_folder))
    shuffle(backgrounds)
    return backgrounds


# In[9]:


alpha_folder = "../alpha_data"
folders = ['train', 'val', 'segmented_test']
augmented_folder = "../augmented_data"
classes = sorted(os.listdir(alpha_folder + '/' +folders[0]))
background_folder = 'backgrounds'
backgrounds = shuffle_backgrounds()   
print(backgrounds[10])

if os.path.exists(augmented_folder):
    shutil.rmtree(augmented_folder)
os.mkdir(augmented_folder)


# In[11]:


def open_background(backgrounds, i):
        chosen_background = backgrounds[i]
        return Image.open(background_folder + '/' + chosen_background)


# In[15]:


def apply_background(i, src, dest):
    for img in os.listdir(src):
        background = open_background(backgrounds, i % len(backgrounds))
        i = i+1
        image = Image.open(src + img)
        background.paste(image, (0,0), image)
        background = background.convert('RGB')
        background.save(dest + img, "PNG")
    return i
        

i = 0
for folder in folders:
    if os.path.exists(augmented_folder + '/' + folder):
        shutil.rmtree(augmented_folder + '/' + folder)
    os.mkdir(augmented_folder + '/' + folder)
    for clss in classes:
        if os.path.exists(augmented_folder+ '/' + folder + '/' + clss):
            shutil.rmtree(augmented_folder+ '/' + folder + '/' + clss)
        dest = augmented_folder+ '/' + folder + '/' + clss + '/'
        os.mkdir(dest)
        src = alpha_folder + '/' + folder + '/' + clss + '/'
        print(src)
        i = apply_background(i, src, dest)


# In[19]:




