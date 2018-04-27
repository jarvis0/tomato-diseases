
# coding: utf-8

# In[12]:


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision
import tqdm
import os
import shutil
from tensorboardX import SummaryWriter
from torch.utils.data.dataset import random_split


# In[13]:


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = 'data/'


# In[14]:


binary_classes = ['healthy', 'non_healthy']


# In[15]:


batch_size = 16
data_transforms = {
        'train': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                    data_transforms[x]) for x in ['train']}
num_folds = 10
splits = []
for i in range(num_folds):
    splits.append('split' + str(i))

train_subtest_dataset_len = len(image_datasets['train'])

splits_size = []
overall_splits_size = 0
for i in range(num_folds - 1):
    overall_splits_size += train_subtest_dataset_len // num_folds
    splits_size.append(train_subtest_dataset_len // num_folds)
splits_size.append(train_subtest_dataset_len - overall_splits_size)

random_splits = random_split(image_datasets['train'], splits_size)
del image_datasets['train']
for i in range(num_folds):
    image_datasets[splits[i]] = random_splits[i]


# In[16]:


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 3 input image channel,
        # 6 output channel,
        # 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation
        self.fc1 = nn.Linear(16 * 61 * 61, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)
        
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


# In[18]:


def build_dataloaders(validation_set):
    datasets = {}
    datasets['train'] = []
    for split in range(num_folds):
        k = 'split' + str(split)
        if split != validation_set:
            datasets['train'] = torch.utils.data.ConcatDataset([datasets['train'], image_datasets[k]])
        else:
            #validation set
            datasets['val'] = image_datasets[k]
            
    dataset_sizes = {x: len(datasets[x]) for x in ['train', 'val']}
    
    # creating dataloaders
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                        shuffle=True, num_workers=4) for x in ['train', 'val']}
    return dataloaders, dataset_sizes


# In[19]:


num_epochs = 10

log_dir = 'log/'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)

folds_performances = {}
for validation_set in range(num_folds):
    
    writer = SummaryWriter(log_dir + 'Fold #' + str(validation_set))
    dataloaders, dataset_sizes = build_dataloaders(validation_set)
    
    model = Model()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    folds_performances[validation_set] = []
    for epoch in tqdm.tqdm(list(range(num_epochs))):  # loop over the dataset multiple epochs
        
        #training and validation part
        for phase in ['train', 'val']:

            if phase == 'train':
                exp_lr_scheduler.step()
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
                
                preds = torch.round(torch.sigmoid(outputs))
                running_corrects += torch.sum(preds.view(-1) == labels.data.float())
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.data.item() / dataset_sizes[phase]
            print('Validation fold #{} - {} Loss: {:.4f} Acc: {:.4f}'.format(validation_set, phase, epoch_loss, epoch_acc))

        folds_performances[validation_set].append(epoch_acc)
        
        # tensorboard statistics
        writer.add_scalar('/loss', epoch_loss, epoch)
        writer.add_scalar('/accuracy', epoch_acc, epoch)
            
print('Training is over.')


# In[20]:


def save_to_file(filename, to_file):
    f = open(filename + '.txt', 'w+')
    f.write(str(to_file))
    f.close()


# In[21]:


for i in range(0, num_folds):
    print(str(i) + ": " + str(folds_performances[i]))

save_to_file('folds_performances', folds_performances)

