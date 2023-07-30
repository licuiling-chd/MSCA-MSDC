import torch
import torch.nn as nn


class MSCA(nn.Module):
    def __init__(self):
        super(MSCA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y1 = self.conv1(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y2 = self.conv2(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y3 = self.conv3(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = y1 + y2 + y3
        y = self.sigmoid(y)
        return x * y.expand_as(x)

def _SplitChannels(channels, num_groups):
    split_channels = [channels//num_groups for _ in range(num_groups)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels

class MSDC(nn.Module):
    def __init__(self, channels, kernel_size):
        super(MSDC, self).__init__()
        self.channels = channels
        self.kernel_num = len(kernel_size)
        self.k = kernel_size
        self.sp = _SplitChannels(channels, self.kernel_num)
        self.conv = nn.ModuleList()
        for i in range(self.kernel_num):
            self.conv.append(nn.Sequential(nn.Conv2d(self.sp[i], 
                                                     self.sp[i], 1),
                                           nn.BatchNorm2d(self.sp[i]),
                                           nn.ReLU(),
                                           nn.Conv2d(self.sp[i],
                                                     self.sp[i],
                                                     (self.k[i], self.k[i]),
                                                     (1, 1),
                                                     ((self.k[i] - 1) // 2, (self.k[i] - 1) // 2),
                                                     groups=self.sp[i], 
                                                     bias=False)))

    def forward(self, x):
        x_split = torch.split(x, self.sp, dim=1)
        xs = []
        for i in range(0, self.kernel_num):
            x = self.conv[i](x_split[i])
            xs.append(x)
            x = torch.cat(xs, dim=1)
        return x


class MSCA_MSDC(nn.Module):
    def __init__(self, bands, classes):
        super(MSCA_MSDC, self).__init__()
        self.att1 = MSCA()
        self.conv1 = nn.Conv2d(in_channels=bands, out_channels=bands, kernel_size=(1, 1))
        self.br1 = nn.Sequential(nn.BatchNorm2d(bands), nn.ReLU())

        self.conv2 = MSDC(bands, [3, 5, 7])
        self.br2 = nn.Sequential(nn.BatchNorm2d(bands), nn.ReLU())

        self.conv3 = MSDC(bands, [3, 5, 7])
        self.br3 = nn.Sequential(nn.BatchNorm2d(bands), nn.ReLU())

        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(bands, classes)
        self.classifier = nn.LogSoftmax(dim=1)

    def forward(self, X):
        # input: (b, 1, d, w, h)
        _, _, d, _, _ = X.size()
        x = X.squeeze(1)
        x = self.att1(x)
        x = self.br1(self.conv1(x))
        x = self.br2(self.conv2(x))
        x = self.br3(self.conv3(x))

        x = self.avg_pooling(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.classifier(x)
        return x
