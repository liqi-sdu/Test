import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.RReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.RReLU())

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1


class mat_net(nn.Module):
    def __init__(self):
        super(mat_net, self).__init__()
        # self.base_model=torchvision.models.resnet18(True)      #预训练模型
        # self.base_layers=list(self.base_model.children())
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.RReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.RReLU())
        self.layer2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.RReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.RReLU())
        self.layer3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.RReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.RReLU())
        self.layer4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.RReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.RReLU())
        self.layer5=nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(256),
            nn.RReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.RReLU())


        # self.layer6=nn.Sequential(
        #    nn.MaxPool2d(kernel_size=2, stride=2),
        #    nn.Conv2d(256,512,kernel_size=3,stride=1,padding=1),
        #    nn.BatchNorm2d(512),
        #    nn.RReLU(),
        #    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
        #    nn.BatchNorm2d(512),
        #    nn.RReLU())
        # self.decode5=Decoder(512,512,256)

        self.decode4 = Decoder(256, 256, 128)
        self.decode3 = Decoder(128, 128, 64)
        self.decode2 = Decoder(64, 64, 32)
        self.decode1 = Decoder(32, 32, 16)
        self.decode0 = nn.Sequential(
            # nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            # nn.Conv2d(8,8,kernel_size=3, padding=1),
            # nn.BatchNorm2d(8),
            # nn.RReLU()
            )
        # self.decode=nn.Sequential(
        #    nn.Conv2d(8,1,kernel_size=3, padding=1),
        #    nn.BatchNorm2d(1),
        #    nn.RReLU(),
        #    nn.Conv2d(1,1,kernel_size=3, padding=1),
        #    nn.BatchNorm2d(1),
        #    nn.ReLU())
    def forward(self, input):
        e1 = self.layer1(input)
        e2 = self.layer2(e1)
        e3 = self.layer3(e2)
        e4 = self.layer4(e3)
        # e5=self.layer5(e4)
        f = self.layer5(e4)
        # d5=self.decode5(f,e4)
        d4 = self.decode4(f,e4)
        d3 = self.decode3(d4, e3)
        d2 = self.decode2(d3, e2)
        d1 = self.decode1(d2, e1)
        d0 = self.decode0(d1)
        # d=self.decode(d0)
        #        d0=d0.view(a*a,1)
        #        output=A.mm(d0)
        output = d0
        return output

