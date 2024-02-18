from sklearn.model_selection import train_test_split
from fastcore.basics import Path, AttrDict
from dataset import SPLID
import torch
from datetime import datetime
import torch.nn as nn
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
from models import UTime

import os

config = AttrDict(
    challenge_data_dir = Path('~/Projects/splid-comp/dataset').expanduser(),
    valid_ratio = 0.1,
    kernel_size = 5,
    tolerance= 6, # Default evaluation tolerance
)

# Define the directory paths
train_data_dir = config.challenge_data_dir / "train_v2"

# Load the ground truth data
ground_truth = config.challenge_data_dir / 'train_labels_v2.csv'

datalist = []

# Searching for training data within the dataset folder
for file in os.listdir(train_data_dir):
    if file.endswith(".csv"):
        datalist.append(os.path.join(train_data_dir, file))

# Sort the training data and labels
datalist = sorted(datalist, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))


train_datalist, test_datalist = train_test_split(datalist, test_size=0.25)


cols = ['Inclination (deg)', 'Longitude (deg)']  # Select cols to pass to ml model

# Train and test datasets/dataloaders for pytorch
trn_data = SPLID(train_datalist, ground_truth, cols)
tst_data = SPLID(test_datalist, ground_truth, cols, classes=trn_data.le_type.classes_)

trn_loader = data.DataLoader(trn_data, shuffle=True, batch_size=10)
tst_loader = data.DataLoader(tst_data, shuffle=True, batch_size=10)

# Training params
lr = 5e-6
n_epochs = 1000
best_tst_loss = 1_000_000.

model = UTime(len(trn_data.le_type.classes_))
model = model.cuda()
criterion = nn.NLLLoss()
opt = torch.optim.Adam(model.parameters(), lr=lr)

print('Start model training')

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
writer = SummaryWriter('runs/splid_trainer_{}'.format(timestamp))

# Training loop
for epoch in range(1, n_epochs + 1):

    print('EPOCH {}:'.format(epoch))

    running_loss = 0.
    last_loss = 0.
    model.train(True)

    for i, (x_batch, y_batch) in enumerate(trn_loader):

        x_batch = x_batch.cuda()
        y_batch = y_batch.cuda()
        opt.zero_grad()
        out = model(x_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        opt.step()

        running_loss += loss.item()
        if i % 50 == 49:
            last_loss = running_loss / 50  # loss per batch
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch * len(trn_loader) + i + 1
            writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0.

    running_tst_loss = 0.0
    model.eval()

    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for i, (x_batch, y_batch) in enumerate(tst_loader):
            x_batch = x_batch.cuda()
            y_batch = y_batch.cuda()
            outputs = model(x_batch)
            tst_loss = criterion(outputs, y_batch)
            running_tst_loss += tst_loss

    avg_tst_loss = running_tst_loss / (i + 1)
    print('LOSS train {} valid {}'.format(last_loss, avg_tst_loss))

    # Log the running loss averaged per batch
    # for both training and validation
    writer.add_scalars('Training vs. Validation Loss',
                       {'Training': last_loss, 'Validation': avg_tst_loss},
                       epoch)
    writer.flush()

    # Track best performance, and save the model's state
    if avg_tst_loss < best_tst_loss:
        best_tst_loss = avg_tst_loss
        model_path = 'model_{}.pth'.format(timestamp)
        torch.save(model.state_dict(), 'saved_models/' + model_path)