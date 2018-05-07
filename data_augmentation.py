
# coding: utf-8

# In[ ]:


from PIL import Image
from matplotlib.pyplot import imshow
import numpy as np
import math
import os
import shutil
import random

#%matplotlib inline


# In[ ]:


def DisplayImage(image):
    imshow(np.asarray(image))


# In[ ]:


binary_classification = False

if binary_classification:
    classes = ['healthy', 'non_healthy']
else:  
    classes = ['healthy', 'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot',
           'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'Tomato_mosaic_virus', 'Tomato_Yellow_Leaf_Curl_Virus']

data_dir = 'augmented_data/'

if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.mkdir(data_dir)

random.seed(a=1234)
shutil.copytree('data/test', data_dir + 'test/')
os.mkdir(data_dir + 'train/')
for clss in classes:
    os.mkdir(data_dir + 'train/' + clss)
    src = 'data/train/' + clss + '/'
    dst = data_dir + 'train/' + clss + '/'
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
        #print(im.format, im.size, im.mode)
        #out = ScaleRotateTranslate(im, 0, scale=(0.8, 0.8))
        #out = im.transpose(Image.TRANSVERSE)#.ROTATE_180)
        
"""center_x = 0
        center_y = 0
        scaling_factor_x = 1
        scaling_factor_y = 1
        shear_x = 0.2
        shear_y = 0
        transform_matrix = (scaling_factor_x, shear_x, center_x,
                              shear_y, scaling_factor_y, center_y)"""
        
"""transform_matrix = (1, 1, 0,
                            0, 1, 0,
                            0, 0, 1)
        out = im.transform(im.size, Image.PERSPECTIVE, transform_matrix, resample=Image.BICUBIC)
        
        DisplayImage(out)
        break"""

"""im = Image.open(src + image)
        im.save(dst + str(counter) + '_ORIGINAL.jpg', 'JPEG')
        
        out = im.transpose(Image.FLIP_TOP_BOTTOM)
        out.save(dst + str(counter) + '_FLIP_TOP_BOTTOM.jpg', 'JPEG')
        
        out = im.transpose(Image.FLIP_LEFT_RIGHT)
        out.save(dst + str(counter) + '_FLIP_LEFT_RIGHT.jpg', 'JPEG')
        
        out = im.transpose(Image.ROTATE_180)
        out.save(dst + str(counter) + '_ROTATE_180.jpg', 'JPEG')
        
        out = im.transpose(Image.ROTATE_90)
        out.save(dst + str(counter) + '_ROTATE_90.jpg', 'JPEG')
        
        out = out.transpose(Image.FLIP_TOP_BOTTOM)
        out.save(dst + str(counter) + '_ROTATE_90_TOP_BOTTOM.jpg', 'JPEG')
        
        out = im.transpose(Image.ROTATE_270)
        out.save(dst + str(counter) + '_ROTATE_270.jpg', 'JPEG')
        
        out = out.transpose(Image.FLIP_TOP_BOTTOM)
        out.save(dst + str(counter) + '_ROTATE_270_TOP_BOTTOM.jpg', 'JPEG')
"""

