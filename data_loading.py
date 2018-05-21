import os
import random
import shutil

DEBUG = False
data_dir = 'data/'
test_size = 30

classes = sorted(os.listdir('original_dataset/'))

if os.path.exists(data_dir):
    shutil.rmtree(data_dir)
os.mkdir(data_dir)
os.mkdir(data_dir + 'train')
os.mkdir(data_dir + 'test')

random.seed(a=1234)
if not DEBUG:
    for cls in list(classes):
        os.mkdir(data_dir + 'train/' + cls)
        os.mkdir(data_dir + 'test/' + cls)
        for image in sorted(os.listdir('original_dataset/' + cls + '/')):
            if random.randint(0, 100) <= test_size:
                shutil.copyfile('original_dataset/' + cls + '/' + image, data_dir + 'test/' + cls + '/' + image)
            else:
                shutil.copyfile('original_dataset/' + cls + '/' + image, data_dir + 'train/' + cls + '/' + image)
        
else:
    count = 0
    for cls in list(classes):
        os.mkdir(data_dir + 'train/' + cls)
        os.mkdir(data_dir + 'test/' + cls)
        for image in sorted(os.listdir('original_dataset/' + cls + '/')):
            if count == 100:
                count = 0
                break
            count += 1
            if random.randint(0, 100) <= test_size:
                shutil.copyfile('original_dataset/' + cls + '/' + image, data_dir + 'test/' + cls + '/' + image)
            else:
                shutil.copyfile('original_dataset/' + cls + '/' + image, data_dir + 'train/' + cls + '/' + image)
