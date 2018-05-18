
# coding: utf-8

# In[7]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import LeNet
import os
from tqdm import trange


# In[8]:


num_epochs = 57
batch_size = 64
lr = 0.001
wd = 0

model_format = ".model"


# In[9]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = '../augmented_data/'


# In[10]:


mean = [0.44947562, 0.46524084, 0.40037745]
std = [0.18456618, 0.16353698, 0.20014246]

data_transforms = {
        'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])}

train_images = datasets.ImageFolder(os.path.join(data_dir, 'train'),
                    data_transforms['train'])

train_dataloader = DataLoader(train_images, batch_size=batch_size, shuffle=True, num_workers=4)


# In[ ]:


model = models.alexnet(pretrained=True)
for param in model.features:
    param.requires_grad = True
model.classifier[6] = nn.Linear(4096, 10)
nn.init.kaiming_normal_(model.classifier[1].weight, nonlinearity='relu')
nn.init.kaiming_normal_(model.classifier[4].weight, nonlinearity='relu')
nn.init.kaiming_normal_(model.classifier[6].weight, nonlinearity='relu')

model.to(device)
model.train()

optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd, eps=0.1)
    
epoch_progress = trange(num_epochs, bar_format='{desc}{r_bar}')
for epoch in epoch_progress:  # loop over the dataset multiple epochs
    
    # iterate over the data
    for inputs, labels in train_dataloader:

        # get the inputs
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad() # zero the gradient buffers

        # forward + loss
        outputs = model(inputs)
        loss = nn.CrossEntropyLoss()(outputs, labels)
                
        # backward + optimize
        loss.backward()
        optimizer.step() # does the update

print('Training is over.')

torch.save(model.state_dict(), 'alexnet_pretrained' + model_format)
print('Model saved to file.')

