import torch.nn as nn
import torch
import torch.nn.functional as F


class UTime(nn.Module):
    def __init__(self, classes):
        super().__init__()
        """ Encoder """
        self.e1 = EncoderBlock(2, 16, 10)
        self.e2 = EncoderBlock(16, 32, 8)
        self.e3 = EncoderBlock(32, 64, 6)
        self.e4 = EncoderBlock(64, 128, 4)
        """ Bottleneck """
        self.b = ConvBlock(128, 256)
        """ Decoder """
        self.d1 = DecoderBlock(256, 128)
        self.d2 = DecoderBlock(128, 64)
        self.d3 = DecoderBlock(64, 32)
        self.d4 = DecoderBlock(32, 16)
        """ Classifier """
        self.outputs = Classifier(16, classes)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        """ Bottleneck """
        b = self.b(p4)
        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)
        """ Classifier """
        outputs = self.outputs(d4)
        return outputs


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, kernel_size=5, padding='same')
        self.conv2 = nn.Conv1d(out_c, out_c, kernel_size=5, padding='same')
        self.bn = nn.BatchNorm1d(out_c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c, pool):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool1d(pool)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=5)
        self.conv = ConvBlock(out_c * 2, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        diff = skip.size()[2] - x.size()[2]
        x = F.pad(
            x, [diff // 2, diff - diff // 2]
        )
        x = torch.cat([skip, x], axis=1)
        x = self.conv(x)
        return x


class Classifier(nn.Module):
    def __init__(self, in_c, classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, classes*2, kernel_size=1, padding='same')
        self.tanh = nn.Tanh()
        self.conv2 = nn.Conv1d(classes*2, classes*2, kernel_size=1, padding='same')
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.tanh(x)
        x = self.conv2(x)
        x = torch.reshape(x, (x.shape[0], x.shape[1]//2, 2, x.shape[2]))
        x = self.softmax(x)
        return x
