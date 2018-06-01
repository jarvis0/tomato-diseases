
# coding: utf-8

# In[4]:


from PIL import Image
import os
import shutil


# In[5]:


folders = ['segmented_test']#['train', 'val']
classes = os.listdir('../augmented_data/val')
data_dir = '../alpha_data/'


# In[6]:


#if os.path.exists(data_dir):
#    shutil.rmtree(data_dir)
#os.mkdir(data_dir)

def f(folder, clss):
    os.mkdir(data_dir+ '/' + folder + '/' + clss)
    src = '../data/'+ folder + '/' + clss + '/'
    print(src)
    for img in os.listdir(src):
        image = Image.open(src + img)
        image = image.convert('RGBA')
        datas = image.getdata()

        new_data = []
        for item in datas:
            if item[0] < 15 and item[1] < 10 and item[2] < 15:
                new_data.append((255, 255, 255, 0))
            else:
                new_data.append(item)

        image.putdata(new_data)
        image.save(data_dir + folder + '/' + clss + '/' + img.replace('.jpg', '.png'), "PNG")

for folder in folders:
    if os.path.exists(data_dir+ '/' + folder):
        shutil.rmtree(data_dir+ '/' + folder)
    os.mkdir(data_dir+ '/' + folder)
    for clss in classes:
        f(folder, clss)
        

