
# coding: utf-8

# In[5]:


from PIL import Image
from matplotlib.pyplot import imshow
import numpy as np
import math
import os
import shutil
import random


# In[6]:


classes = os.listdir('data/train/')
destination = 'augmented_data/'

if os.path.exists(destination):
    shutil.rmtree(destination)
os.mkdir(destination)

random.seed(a=1234)
os.mkdir(destination + 'train/')
os.mkdir(destination + 'val/')
for s in ['train', 'val']:
    for clss in classes:
        os.mkdir(destination + s + '/' + clss)
        src = 'data/' + s + '/' + clss + '/'
        dst = destination + s + '/' + clss + '/'
        counter = 0
        for image in os.listdir(src):
            im = Image.open(src + image)
            im.save(dst + str(counter) + '_ORIGINAL.jpg', 'JPEG')

            transf = random.randint(1, 4)

            if transf == 1:
                out = im.transpose(Image.FLIP_TOP_BOTTOM)
                out.save(dst + str(counter) + '_FLIP_TOP_BOTTOM.jpg', 'JPEG')

                out = im.transpose(Image.ROTATE_90)
                out.save(dst + str(counter) + '_ROTATE_90.jpg', 'JPEG')

            if transf == 2:
                out = im.transpose(Image.FLIP_LEFT_RIGHT)
                out.save(dst + str(counter) + '_FLIP_LEFT_RIGHT.jpg', 'JPEG')

                out = im.transpose(Image.ROTATE_180)
                out.save(dst + str(counter) + '_ROTATE_180.jpg', 'JPEG')

            if transf == 3:
                out = im.transpose(Image.ROTATE_90)
                out = out.transpose(Image.FLIP_TOP_BOTTOM)
                out.save(dst + str(counter) + '_ROTATE_90_TOP_BOTTOM.jpg', 'JPEG')

                out = im.transpose(Image.ROTATE_270)
                out = out.transpose(Image.FLIP_TOP_BOTTOM)
                out.save(dst + str(counter) + '_ROTATE_270_TOP_BOTTOM.jpg', 'JPEG')

            if transf == 4:
                out = im.transpose(Image.ROTATE_270)
                out.save(dst + str(counter) + '_ROTATE_270.jpg', 'JPEG')

                out = im.transpose(Image.FLIP_LEFT_RIGHT)
                out.save(dst + str(counter) + '_FLIP_LEFT_RIGHT.jpg', 'JPEG')

            counter += 1

