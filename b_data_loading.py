import os
import random
import shutil

DEBUG = False
source = 'original_segmented_dataset/'
destination = 'data/'
test_size = 30
validation_size = 14

classes = ['Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot',
           'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'Tomato_mosaic_virus', 'Tomato_Yellow_Leaf_Curl_Virus', 'healthy']

if os.path.exists(destination):
    shutil.rmtree(destination)
os.mkdir(destination)
os.mkdir(destination + 'train')
os.mkdir(destination + 'val')
os.mkdir(destination + 'segmented_test')
os.mkdir(destination + 'normal_test')
for cls in list(classes):
    os.mkdir(destination + 'train/' + cls)
    os.mkdir(destination + 'val/' + cls)
    os.mkdir(destination + 'segmented_test/' + cls)
    os.mkdir(destination + 'normal_test/' + cls)

random.seed(a=1234)
if not DEBUG:
    for cls in list(classes):
        for image in os.listdir(source + cls + '/'):
            r = random.randint(0, 100)
            if r <= test_size:
                shutil.copyfile(source + cls + '/' + image, destination + 'segmented_test/' + cls + '/' + image)
                shutil.copyfile('original_dataset/' + cls + '/' + image.replace('_final_masked', ''), destination + 'normal_test/' + cls + '/' + image)
            elif r <= test_size + validation_size:
                shutil.copyfile(source + cls + '/' + image, destination + 'val/' + cls + '/' + image)
            else:
                shutil.copyfile(source + cls + '/' + image, destination + 'train/' + cls + '/' + image)
        
else:
    count = 0
    for cls in list(classes):
        for image in os.listdir(source + cls + '/'):
            if count == 100:
                count = 0
                break
            count += 1
            r = random.randint(0, 100)
            if r <= test_size:
                shutil.copyfile(source + cls + '/' + image, destination + 'segmented_test/' + cls + '/' + image)
                shutil.copyfile('original_dataset/' + cls + '/' + image.replace('_final_masked', ''), destination + 'normal_test/' + cls + '/' + image)
            elif r <= test_size + validation_size:
                shutil.copyfile(source + cls + '/' + image, destination + 'val/' + cls + '/' + image)
            else:
                shutil.copyfile(source + cls + '/' + image, destination + 'train/' + cls + '/' + image)
