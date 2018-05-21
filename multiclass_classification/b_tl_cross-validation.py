
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import trange
import os
import argparse
from tensorboardX import SummaryWriter
import time
import datetime


# In[ ]:


def save_to_file(filename, to_file):
    f = open(log_dir + filename, 'a')
    f.write(str(to_file))
    f.close()


# In[ ]:


ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d h%Hm%Ms%S')

if not os.path.exists('log/'):
    os.mkdir('log/')
    
log_dir = 'log/' + timestamp + '/'
os.mkdir(log_dir)


# In[ ]:


parser = argparse.ArgumentParser(description='CNN hyperparameters.')
parser.add_argument('--arc', dest='arc', default='AlexNet_pretrained', type=str, required=False)
parser.add_argument('--data', dest='data', default='segmented', type=str, required=False)
parser.add_argument('--num_epochs', dest='num_epochs', default=60, type=int, required=False)
parser.add_argument('--batch_size', dest='batch_size', default=64, type=int, required=False)
parser.add_argument('--lr', dest='lr', default=0.001, type=float, required=False)
parser.add_argument('--wd', dest='wd', default=0, type=float, required=False)

args = parser.parse_args()
arc = args.arc
d = args.data
num_epochs = args.num_epochs
batch_size = args.batch_size
lr = args.lr
wd = args.wd
"""
num_epochs = 2
batch_size = 64
lr = 0.001
wd = 0
"""
infos = {}
infos['architecture'] = arc
infos['dataset'] = d
infos['num_epochs'] = num_epochs
infos['batch_size'] = batch_size
infos['lr'] = lr
infos['wd'] = wd
save_to_file('infos', infos)


# In[ ]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = '../augmented_data/'
model_format = ".model"


# In[ ]:


mean = [0.415664, 0.44468355, 0.25746378]
std = [0.24116626, 0.22172716, 0.20130461]

data_transforms = {
        'train': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
        'val': transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                    data_transforms[x]) for x in ['train', 'val']}


# In[ ]:


def build_dataloaders(datasets):
    
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    
    # creating dataloaders
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size,
                        shuffle=True, num_workers=4) for x in ['train', 'val']}
    
    return dataloaders, dataset_sizes

dataloaders, dataset_sizes = build_dataloaders(image_datasets)

conf_matrices = {}
performances = []
writer = SummaryWriter(log_dir)


# In[ ]:


model = models.alexnet(pretrained=True)


# In[ ]:


for param in model.features:
    param.requires_grad = True

model.classifier[6] = nn.Linear(4096, 10)

nn.init.kaiming_normal_(model.classifier[1].weight, nonlinearity='relu')
nn.init.kaiming_normal_(model.classifier[4].weight, nonlinearity='relu')
nn.init.kaiming_normal_(model.classifier[6].weight, nonlinearity='relu')

model.to(device)


# In[ ]:


optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd, eps=0.1)
    
epoch_progress = trange(num_epochs, desc='Epoch 0 - val loss: ? acc: ?', bar_format='{desc}{r_bar}')
for epoch in epoch_progress:  # loop over the dataset multiple epochs
        
    conf_matrices[epoch] = {}
    # training and validation part
    for phase in ['train', 'val']:
            
        if phase == 'train':
            model.train()  # Set to training mode
        else:
            model.eval()  # Set model to evaluate mode
                
        running_loss = 0.0
        running_corrects = 0
        conf_matrices[epoch] = [0] * 10
        for i in range(10):
            conf_matrices[epoch][i] = [0] * 10

        # iterate over the data
        for inputs, labels in dataloaders[phase]:
            
            # get the inputs
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # zero the gradient buffers

            # forward + loss
            outputs = model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
                
            if phase == 'train':
                # backward + optimize
                loss.backward()
                optimizer.step() # does the update

            # statistics
            running_loss += loss.data.item() * inputs.size(0)
                
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
                
            if phase == 'val':
                for predicted in range(len(preds)):
                    conf_matrices[epoch][labels.data[predicted]][preds[predicted]] += 1
        
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.data.item() / dataset_sizes[phase]
            
        # tensorboard statistics
        if phase == 'train':
            writer.add_scalar('/train_loss', epoch_loss, epoch)
            writer.add_scalar('/train_accuracy', epoch_acc, epoch)
        else:
            writer.add_scalar('/val_loss', epoch_loss, epoch)
            writer.add_scalar('/val_accuracy', epoch_acc, epoch)
    
    performances.append(epoch_acc)
    epoch_progress.set_description('Epoch {} - val loss: {:.4f} acc: {:.4f}'.format(epoch, epoch_loss, epoch_acc), refresh=False)
    
print('Training is over.')


# In[ ]:


torch.save(model.state_dict(), log_dir + 'alexnet_pretrained' + model_format)
print('Model saved to file.')


# In[ ]:


save_to_file('performances', performances)

for epoch in range(num_epochs):
    conf_matrices[str(epoch)] = conf_matrices[epoch]
    del conf_matrices[epoch]
save_to_file('confusion_matrices', conf_matrices)

print('Statistics saved to file.')

