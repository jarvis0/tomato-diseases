import os
import random
import shutil

DEBUG = True
data_dir = 'data/'
test_size = 20

classes = ['healthy', 'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot',
           'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'Tomato_mosaic_virus', 'Tomato_Yellow_Leaf_Curl_Virus']

if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.mkdir(data_dir)
os.mkdir(data_dir + 'train')
os.mkdir(data_dir + 'test')
os.mkdir(data_dir + 'train/healthy')
os.mkdir(data_dir + 'train/non_healthy')
os.mkdir(data_dir + 'test/healthy')
os.mkdir(data_dir + 'test/non_healthy')

random.seed(a=1234)
for image in os.listdir('original_dataset/healthy/'):
    if random.randint(1, 100) <= test_size:
        shutil.copyfile('original_dataset/healthy/' + image, data_dir + 'test/healthy/' + image)
    else:
        shutil.copyfile('original_dataset/healthy/' + image, data_dir + 'train/healthy/' + image)

if not DEBUG:
    for cls in list(classes[1:]):
        for image in os.listdir('original_dataset/' + cls + '/'):
            if random.randint(1, 100) <= test_size:
                shutil.copyfile('original_dataset/' + cls + '/' + image, data_dir + 'test/non_healthy/' + image)
            else:
                shutil.copyfile('original_dataset/' + cls + '/' + image, data_dir + 'train/non_healthy/' + image)
else:
    count = 0
    for cls in list(classes[1:]):
        for image in os.listdir('original_dataset/' + cls + '/'):
            if count == 175:
                count = 0
                break
            count += 1
            if random.randint(1, 100) <= test_size:
                shutil.copyfile('original_dataset/' + cls + '/' + image, data_dir + 'test/non_healthy/' + image)
            else:
                shutil.copyfile('original_dataset/' + cls + '/' + image, data_dir + 'train/non_healthy/' + image)
