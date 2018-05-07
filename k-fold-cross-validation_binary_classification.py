
# coding: utf-8

# In[2]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
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


# In[2]:


def save_to_file(filename, to_file):
    f = open(log_dir + filename, 'a+')
    f.write(str(to_file))
    f.close()


# In[3]:


ts = time.time()
timestamp = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %Hh%Mm%Ss')

if not os.path.exists('log/'):
    os.mkdir('log/')
    
log_dir = 'log/' + timestamp + '/'
os.mkdir(log_dir)


# In[4]:


parser = argparse.ArgumentParser(description='CNN hyperparameters.')
parser.add_argument('--num_epochs', dest='num_epochs', default=20, type=int, required=False)
parser.add_argument('--batch_size', dest='batch_size', default=16, type=int, required=False)
parser.add_argument('--lr', dest='lr', default=0.001, type=float, required=False)
parser.add_argument('--wd', dest='wd', default=0, type=float, required=False)

args = parser.parse_args()
num_epochs = args.num_epochs
batch_size = args.batch_size
lr = args.lr
wd = args.wd
save_to_file('infos', 'num_epochs={}, batch_size={}, lr={}, wd={}'.format(num_epochs, batch_size, lr, wd))


# In[5]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = 'data/'


# In[6]:


mean = [0.44947562, 0.46524084, 0.40037745]
std = [0.18456618, 0.16353698, 0.20014246]

data_transforms = {
        'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                    data_transforms[x]) for x in ['train']}
num_folds = 10
splits = []
for i in range(num_folds):
    splits.append('split' + str(i))

train_dataset_len = len(image_datasets['train'])

splits_size = []
overall_splits_size = 0
for i in range(num_folds - 1):
    overall_splits_size += train_dataset_len // num_folds
    splits_size.append(train_dataset_len // num_folds)
splits_size.append(train_dataset_len - overall_splits_size)

random_splits = random_split(image_datasets['train'], splits_size)
del image_datasets['train']
for i in range(num_folds):
    image_datasets[splits[i]] = random_splits[i]


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


# In[ ]:


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


# In[ ]:


folds_performances = {}
for validation_set in trange(num_folds, desc='Folds iterations: '):
    
    writer = SummaryWriter(log_dir + 'Fold ' + str(validation_set))
    dataloaders, dataset_sizes = build_dataloaders(validation_set)
    
    model = Model()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    #optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    folds_performances[validation_set] = []
    epoch_progress = trange(num_epochs, desc='Fold {}, epoch 0 - val loss: ? acc: ?'.format(validation_set))
    for epoch in epoch_progress:  # loop over the dataset multiple epochs

        #training and validation part
        for phase in ['train', 'val']:
            
            if phase == 'train':
                #exp_lr_scheduler.step()
                model.train()  # Set to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # iterate over the data
            for inputs, labels in dataloaders[phase]:

                # get the inputs
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad() # zero the gradient buffers

                # forward + loss
                outputs = model(inputs)
                loss = nn.BCEWithLogitsLoss()(outputs.view(-1), labels.float())
                
                if phase == 'train':
                    # backward + optimize
                    loss.backward()
                    optimizer.step() # does the update

                # statistics
                running_loss += loss.data.item() * inputs.size(0)
                
                preds = torch.round(F.sigmoid(outputs))
                running_corrects += torch.sum(preds.view(-1) == labels.data.float())
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.data.item() / dataset_sizes[phase]

        folds_performances[validation_set].append(epoch_acc)
        epoch_progress.set_description('Fold {}, epoch {} - val loss: {:.4f} acc: {:.4f}'.format(
                validation_set, epoch, epoch_loss, epoch_acc), refresh=False)

        # tensorboard statistics
        writer.add_scalar('/loss', epoch_loss, epoch)
        writer.add_scalar('/accuracy', epoch_acc, epoch)

print('K-fold cross-validation is over.')


# In[ ]:


save_to_file('folds_performances', folds_performances)
# Statistical approach
total_epoch_performace = [0] * num_epochs
for epoch in range(0, num_epochs):
    for i in range (0, num_folds):
        total_epoch_performace[epoch] += folds_performances[i][epoch]

# Mitchell approach
total_max_performance = [0] * num_folds
for i in folds_performances:
    total_max_performance[i] = folds_performances[i].index(max(folds_performances[i]))

save_to_file('infos', '\nStatistical best number of epochs: {}'.format(
                                                                total_epoch_performace.index(max(total_epoch_performace))))
save_to_file('infos', '\nMitchell average best number of epochs: {}'.format(
                                                                sum(total_max_performance) / num_folds))

