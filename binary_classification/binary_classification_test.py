
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
from tqdm import trange


# In[2]:


num_epochs = 15
batch_size = 32
lr = 0.001
wd = 0

binary_classification = True
if binary_classification:
    classes = ['healthy', 'non_healthy']
else:  
    classes = ['healthy', 'Bacterial_spot', 'Early_blight', 'Late_blight', 'Leaf_Mold', 'Septoria_leaf_spot',
           'Spider_mites Two-spotted_spider_mite', 'Target_Spot', 'Tomato_mosaic_virus', 'Tomato_Yellow_Leaf_Curl_Virus']


# In[3]:


def save_to_file(filename, to_file):
    f = open(filename, 'a+')
    f.write(str(to_file))
    f.close()


# In[4]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = '../augmented_data/'


# In[5]:


mean = [0.44947562, 0.46524084, 0.40037745]
std = [0.18456618, 0.16353698, 0.20014246]

data_transforms = {
        'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
        'test': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                    data_transforms[x]) for x in ['train', 'test']}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4) for x in ['train', 'test']}


# In[6]:


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 3 input image channel,
        # 6 output channel,
        # 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)
        
    def forward(self, x):
        # max pooling over a (2, 2) windows
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


# In[9]:


model = Model()
model.to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
epoch_progress = trange(num_epochs, bar_format='{desc}{r_bar}')
for epoch in epoch_progress:  # loop over the dataset multiple epochs
    
    # iterate over the data
    for inputs, labels in dataloaders['train']:

        # get the inputs
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() # zero the gradient buffers

        # forward + loss
        outputs = model(inputs)
        loss = nn.BCEWithLogitsLoss()(outputs.view(-1), labels.float())
                
        # backward + optimize
        loss.backward()
        optimizer.step() # does the update

torch.save(model.state_dict(), 'model')
print('Training is over.')


# In[15]:


model.eval()

correct = 0
total = 0
with torch.no_grad():
    for data in dataloaders['test']:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        total += labels.size(0)
        preds = torch.round(F.sigmoid(outputs))
        correct += torch.sum(preds.view(-1) == labels.data.float())

save_to_file('test_performances', 'Accuracy of the network on the test images: {} %%'.format((100 * correct / total)))


# In[16]:


class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))
with torch.no_grad():
    for data in dataloaders['test']:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        preds = torch.round(F.sigmoid(outputs))
        c = (preds.view(-1) == labels.data.float()).squeeze()
        for i in range(len(classes)):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

for i in range(len(classes)):    
    save_to_file('test_performances', '\nAccuracy of {}: {} %'.format(classes[i], 100 * class_correct[i] / class_total[i]))

