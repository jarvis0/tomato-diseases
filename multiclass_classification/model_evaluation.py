
# coding: utf-8

# In[6]:


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from torch.utils.data import TensorDataset
import torchvision
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import os


# In[7]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = '../data/'
test_dir = 'normal_test/'
classes = sorted(os.listdir(data_dir + test_dir))
batch_size = 64


# In[8]:


# segmented_test normalization
#mean = [0.14318287, 0.19182085, 0.10939839]
#std = [0.20657195, 0.25984347, 0.1585114]

# normal_test normalization
#mean = [0.44947562, 0.46524084, 0.40037745]
#std = [0.18456618, 0.16353698, 0.20014246]

data_transforms = {
        'test': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
        ])}

test_images = datasets.ImageFolder(os.path.join(data_dir, test_dir),
                    data_transforms['test'])

test_dataloader = DataLoader(test_images, batch_size=batch_size, shuffle=False, num_workers=4)


# In[9]:


data = []
for images, labels in test_dataloader:
    for i in range(len(images)):
        numpy_image = images[i].numpy()
        image_mean = np.mean(numpy_image, axis=(1,2))
        std = np.std(numpy_image, axis=(1,2))
        adjusted_stddev = np.float32(np.maximum(std, [1.0/224.0, 1.0/224.0, 1.0/224.0]))
        image_mean_matrix = np.asarray([np.full((224,224), image_mean[0]), np.full((224,224), image_mean[1]), np.full((224,224), image_mean[2])])
        adjusted_stddev_matrix = np.asarray([np.full((224,224), adjusted_stddev[0]), np.full((224,224), adjusted_stddev[1]), np.full((224,224), adjusted_stddev[2])])
        images[i] = images[i].sub_(torch.from_numpy(image_mean_matrix))
        images[i] = images[i].div_(torch.from_numpy(adjusted_stddev_matrix))
            
    data.append(TensorDataset(images, labels))


# In[10]:


datasets = []
for d in data:
    datasets = ConcatDataset([datasets, d])
    
test_dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=False, num_workers=4)


# In[34]:


model = models.alexnet()
model.classifier[6] = nn.Linear(4096, 10)
model.load_state_dict(torch.load('alexnet_pretrained.model', map_location=str(device)))
model.eval()
model.to(device)


# In[35]:


import matplotlib.pyplot as plt
import numpy as np

def imshow(inp):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.pause(0.001)


# In[36]:


DEBUG = False

conf_matrix = {}
conf_matrix = [0] * len(classes)
for i in range(len(classes)):
    conf_matrix[i] = [0] * len(classes)
running_corrects = 0
with torch.no_grad():
    for data in test_dataloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        for predicted in range(len(preds)):
            conf_matrix[labels.data[predicted]][preds[predicted]] += 1
        if DEBUG == True:
            for i in range(len(preds)):
                if labels.data[predicted] != preds[predicted]:
                    print('True class: {}'.format(classes[labels.data[predicted]]))
                    print('Predicted class: {}'.format(classes[preds[predicted]]))
                    out = torchvision.utils.make_grid(images[i])
                    imshow(out)


# In[37]:


def print_to_file(to_file, filename='model_evaluation'):
    f = open(filename, 'a')
    f.write(to_file)
    f.close()


# In[38]:


num_classes = len(classes)
for i in range(num_classes):
    print_to_file(str(conf_matrix[i]).replace('[', '').replace(']', '') + '\n')


# In[39]:


num_samples = 0
for i in range(num_classes):
    for j in range(num_classes):
        num_samples += conf_matrix[i][j]
print_to_file('Number of samples: {}'.format(num_samples))


# In[40]:


accuracy = 0
for i in range(num_classes):
    accuracy += conf_matrix[i][i]
accuracy /= num_samples

print_to_file('\nAccuracy: {}'.format(accuracy))


# In[41]:


macro_precision = 0
precision = [0] * num_classes
precision_total = [0] * num_classes
for i in range(num_classes):
    precision[i] = conf_matrix[i][i]
    for j in range(num_classes):
        precision_total[i] += conf_matrix[j][i]
    if precision_total[i] != 0:
        precision[i] /= precision_total[i]
    else:
        precision[i] = float('NaN')
    macro_precision += precision[i] / num_classes

print_to_file('\nMacro precision: {}'.format(macro_precision))


# In[42]:


macro_recall = 0
recall = [0] * num_classes
recall_total = [0] * num_classes
for i in range(num_classes):
    recall[i] = conf_matrix[i][i]
    for j in range(num_classes):
        recall_total[i] += conf_matrix[i][j]
    if recall_total[i] != 0:
        recall[i] /= recall_total[i]
    else:
        recall[i] = float('NaN')
    macro_recall += recall[i] / num_classes

print_to_file('\nMacro recall (Trues Rate): {}'.format(macro_recall))


# In[43]:


macro_fr = 0
fr_total = [0] * num_classes
fr = [0] * num_classes
for i in range(num_classes):
    for j in range(num_classes):
        fr_total[i] += conf_matrix[i][j]
    fr[i] = fr_total[i] - conf_matrix[i][i]
    fr[i] /= fr_total[i]
    macro_fr += fr[i] / num_classes

print_to_file('\nFalses Rate: {}'.format(macro_fr))


# In[44]:


macro_f = 0
f = [0] * num_classes

for i in range(num_classes):
    if precision[i] + recall[i] != 0:
        f[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i])
    else:
        f[i] = float('NaN')
    macro_f += f[i] / num_classes

print_to_file('\nMacro F1: {}'.format(macro_f))


# In[45]:


print('Analysis measures saved to file.')

