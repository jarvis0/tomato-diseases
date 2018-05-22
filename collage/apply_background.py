
# coding: utf-8

# In[5]:


from PIL import Image
import os
import random
import shutil


# In[6]:


alpha_folder = "../alpha_data"
folders = ['train', 'val']
augmented_folder = "../augmented_data"
classes = sorted(os.listdir(alpha_folder + '/' + folders[0]))
background_folder = 'backgrounds'

if os.path.exists(augmented_folder):
    shutil.rmtree(augmented_folder)
os.mkdir(augmented_folder)


# In[7]:


def apply_background(src, dest):
    for img in os.listdir(src):
        background = Image.open(background_folder + '/' + random.choice(os.listdir(background_folder)))
        image = Image.open(src + img)
        background.paste(image, (0,0), image)
        background = background.convert('RGB')
        background.save(dest + img, "PNG")
        
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
        apply_background(src, dest)

