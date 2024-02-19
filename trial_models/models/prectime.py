import torch
from torch import nn


class PrecTime(nn.Module):
    def __init__(self, out_classes, n_win, l_win, c_in, c_conv, c_lstm=200, lstm_layers=2):
        super(PrecTime, self).__init__()

        self.feat1 = FeatureExtraction(c_in, c_conv, dilation=1)
        self.feat2 = FeatureExtraction(c_in, c_conv, dilation=4)
        self.windows = n_win

        self.lstm = nn.LSTM(input_size=int(c_conv * l_win), hidden_size=c_lstm, num_layers=lstm_layers,
                            batch_first=True, bidirectional=True)
        self.up1 = nn.Upsample(size=n_win * l_win // 2)
        self.refine = Refinement(2*(c_lstm + c_conv), c_conv, out_classes)

        # Intermediate label prediction steps
        self.fc = nn.Linear(in_features=2*c_lstm, out_features=2*out_classes)
        self.up2 = nn.Upsample(scale_factor=l_win)

    def forward(self, x):
        x1 = self.feat1(x)
        x2 = self.feat2(x)
        f_stack = torch.cat([x1, x2], dim=1)
        x = torch.permute(f_stack, (0, 2, 1))
        x = torch.reshape(x, (x.shape[0], self.windows, -1))
        coarse, (_, _) = self.lstm(x)
        x = torch.permute(coarse, (0, 2, 1))
        x = self.up1(x)
        x = torch.cat([x, f_stack], dim=1)
        x = self.refine(x)
        fine_out = torch.reshape(x, (x.shape[0], x.shape[1] // 2, 2, x.shape[2]))

        x_c = self.fc(coarse)
        x_c = torch.permute(x_c, (0, 2, 1))
        x_c = self.up2(x_c)
        coarse_out = torch.reshape(x_c, (x_c.shape[0], x_c.shape[1] // 2, 2, x_c.shape[2]))

        return fine_out, coarse_out


class FeatureExtraction(nn.Module):
    def __init__(self, c_in, c_conv, dilation):
        super(FeatureExtraction, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=c_in, out_channels=c_conv, kernel_size=5, dilation=dilation, padding='same')
        self.pool = nn.MaxPool1d(2)
        self.relu1 = nn.ReLU()
        self.drop = nn.Dropout1d(0.5)
        self.triconv = nn.Sequential(
            nn.Conv1d(in_channels=c_conv, out_channels=c_conv, kernel_size=5, dilation=dilation, padding='same'),
            nn.Conv1d(in_channels=c_conv, out_channels=c_conv, kernel_size=5, dilation=dilation, padding='same'),
            nn.Conv1d(in_channels=c_conv, out_channels=c_conv, kernel_size=5, dilation=dilation, padding='same')
        )
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.relu1(x)
        x = self.drop(x)
        x = self.triconv(x)
        x = self.relu2(x)
        return x


class Refinement(nn.Module):
    def __init__(self, c_in, c_conv, out_classes):
        super(Refinement, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=c_in, out_channels=c_conv, kernel_size=5, padding='same')
        self.relu1 = nn.ReLU()
        self.up = nn.Upsample(scale_factor=2)
        self.drop = nn.Dropout1d(0.5)
        self.conv2 = nn.Conv1d(in_channels=c_conv, out_channels=c_conv, kernel_size=5, padding='same')
        self.fc = nn.Linear(in_features=c_conv, out_features=2*out_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.up(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.fc(x)
        return torch.permute(x, (0, 2, 1))
