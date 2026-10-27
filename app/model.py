"""ResNet9 architecture extracted from the original notebook, without training code."""
from torch import nn


def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class ResNet9(nn.Module):
    def __init__(self, in_channels=3, num_classes=5):
        super().__init__()
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(nn.AdaptiveMaxPool2d(1), nn.Flatten(),
                                        nn.Dropout(0.2), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.res1(x) + x
        x = self.conv4(self.conv3(x))
        x = self.res2(x) + x
        return self.classifier(x)
