
# coding: utf-8

# In[24]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from tqdm import trange
import os
import argparse
from tensorboardX import SummaryWriter
import time
import datetime
from models import AlexNet


# In[25]:


def save_to_file(filename, to_file):
    f = open(log_dir + filename, 'a')
    f.write(str(to_file))
    f.close()


# In[26]:


ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d h%Hm%Ms%S')

if not os.path.exists('log/'):
    os.mkdir('log/')
    
log_dir = 'log/' + timestamp + '/'
os.mkdir(log_dir)


# In[35]:


parser = argparse.ArgumentParser(description='CNN hyperparameters.')
parser.add_argument('--num_epochs', dest='num_epochs', default=57, type=int, required=False)
parser.add_argument('--batch_size', dest='batch_size', default=64, type=int, required=False)
parser.add_argument('--lr', dest='lr', default=0.0005, type=float, required=False)
parser.add_argument('--wd', dest='wd', default=0, type=float, required=False)

args = parser.parse_args()
num_epochs = args.num_epochs
batch_size = args.batch_size
lr = args.lr
wd = args.wd
"""
num_epochs = 2
batch_size = 32
lr = 0.001
wd = 0
"""
infos = {}
infos['num_epochs'] = num_epochs
infos['batch_size'] = batch_size
infos['lr'] = lr
infos['wd'] = wd
save_to_file('infos', infos)


# In[29]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = '../augmented_data/'


# In[30]:


mean = [0.44947562, 0.46524084, 0.40037745]
std = [0.18456618, 0.16353698, 0.20014246]

data_transforms = {
        'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                    data_transforms[x]) for x in ['train']}
num_folds = 5
splits = []
for k in range(num_folds):
    splits.append('split' + str(k))

train_dataset_len = len(image_datasets['train'])

splits_size = []
overall_splits_size = 0
for k in range(num_folds - 1):
    overall_splits_size += train_dataset_len // num_folds
    splits_size.append(train_dataset_len // num_folds)
splits_size.append(train_dataset_len - overall_splits_size)

random_splits = random_split(image_datasets['train'], splits_size)
del image_datasets['train']
for k in range(num_folds):
    image_datasets[splits[k]] = random_splits[k]


# In[31]:


def build_dataloaders(validation_set):
    datasets = {}
    datasets['train'] = []
    for split in range(num_folds):
        k = 'split' + str(split)
        if split != validation_set:
            datasets['train'] = ConcatDataset([datasets['train'], image_datasets[k]])
        else:
            # validation set
            datasets['val'] = image_datasets[k]
            
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    
    # creating dataloaders
    dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size,
                        shuffle=True, num_workers=4) for x in ['train', 'val']}
    
    return dataloaders, dataset_sizes


# In[32]:


validation_set = 0
dataloaders, dataset_sizes = build_dataloaders(validation_set)

conf_matrices = {}
performances = []
writer = SummaryWriter(log_dir)


# In[33]:


model = AlexNet()
model.to(device)
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

torch.save(model.state_dict(), 'alexnet_model')
print('Model saved to file.')


# In[34]:


save_to_file('performances', performances)

for epoch in range(num_epochs):
    conf_matrices[str(epoch)] = conf_matrices[epoch]
    del conf_matrices[epoch]
save_to_file('confusion_matrices', conf_matrices)

print('Statistics saved to file.')

