
# coding: utf-8

# In[2]:


from PIL import Image
import numpy as np
import os


# In[12]:


data_dir = 'original_segmented_dataset/'
classes = sorted(os.listdir(data_dir))


# In[15]:


leaf = {}
leaf['r'] = []
leaf['g'] = []
leaf['b'] = []
for clss in classes:
    src = data_dir + clss + '/'
    for image in os.listdir(src):
        im = Image.open(src + image)
        for item in im.getdata():
            if not(item[0] < 15 and item[1] < 10 and item[2] < 15):
                leaf['r'].append(item[0])
                leaf['g'].append(item[1])
                leaf['b'].append(item[2])


# In[ ]:


print([np.asarray((leaf['r'])).mean(), np.asarray((leaf['g'])).mean(), np.asarray((leaf['b'])).mean()])
print([np.asarray((leaf['r'])).std(), np.asarray((leaf['g'])).std(), np.asarray((leaf['b'])).std()])

