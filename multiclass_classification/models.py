
# coding: utf-8

# In[6]:


import torch
import torch.nn as nn
import torch.nn.functional as F


# In[7]:


class LeNet(nn.Module):
    
    def __init__(self):
        super(LeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * 61 * 61, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10)
        )
        nn.init.kaiming_normal_(self.features[0].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.features[3].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.classifier[0].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.classifier[2].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.classifier[4].weight, nonlinearity='relu')
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.num_flat_features(x))
        x = self.classifier(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
    @property
    def get_features(self):
        return self.features
    


# In[8]:


class AlexNet(nn.Module):

    def __init__(self):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 10),
        )
        nn.init.kaiming_normal_(self.features[0].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.features[3].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.features[6].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.features[8].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.features[10].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.classifier[1].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.classifier[4].weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.classifier[6].weight, nonlinearity='relu')

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.num_flat_features(x))
        x = self.classifier(x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
        
    @property
    def get_features(self):
        return self.features
    
    @property
    def get_classifier(self):
        return self.classifier
    

