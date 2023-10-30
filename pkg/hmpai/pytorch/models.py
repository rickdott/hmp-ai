from torch import nn


class SAT1Base(nn.Module):
    def __init__(self, n_channels, n_samples, n_classes):
        super().__init__()
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=37632, out_features=128)
        self.linear_final = nn.Linear(in_features=128, out_features=n_classes)
        # Kernel order = (channels, samples)
        self.maxpool = nn.MaxPool2d((1, 2))
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 5))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3))
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(1, 3))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.flatten(x)
        # print(x.shape)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear_final(x)

        return x
