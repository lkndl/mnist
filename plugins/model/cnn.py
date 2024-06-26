import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, n_classes, in_features):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_features, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, n_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.fc1(x)
        return x
