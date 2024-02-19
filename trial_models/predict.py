import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import SPLID
from trial_models.models.prectime import PrecTime
import helpers.utils as utils
import helpers.evaluation as evaluation
from fastcore.basics import Path, AttrDict

config = AttrDict(
    challenge_data_dir = Path('~/Projects/splid-comp/dataset').expanduser(),
    valid_ratio = 0.1,
    kernel_size = 5,
    tolerance = 6, # Default evaluation tolerance
)

# Define the directory paths
train_data_dir = config.challenge_data_dir / "train_v2"
ground_truth = config.challenge_data_dir / 'train_labels_v2.csv'

datalist = []

# Searching for training data within the dataset folder
for file in os.listdir(train_data_dir):
    if file.endswith(".csv"):
        datalist.append(os.path.join(train_data_dir, file))

# Sort the training data and labels
datalist = sorted(datalist, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))
classes = ['ES-ES', 'SS-CK', 'SS-EK', 'SS-HK', 'SS-NK', 'IK-CK', 'IK-EK', 'IK-HK', 'ID-NK', 'AD-NK']
cols = ['Inclination (deg)', 'Longitude (deg)', 'Eccentricity', 'Semimajor Axis (m)', 'RAAN (deg)', 'Argument of Periapsis (deg)', 'Vz (m/s)']

splid = SPLID(datalist, ground_truth, cols, classes=classes)

loader = DataLoader(splid, batch_size=10)
model = PrecTime(len(classes), n_win=92, l_win=24, c_in=len(cols), c_conv=128).cuda()
model.load_state_dict(torch.load('saved_models/model_20240220_113758.pth'))
model.eval()

frames = []

with torch.no_grad():
    for idx, (data, target) in enumerate(loader):
        fine_out, coarse_out = model(data.cuda())
        preds = np.argmax(fine_out.cpu(), axis=1)
        for i in range(preds.shape[0]):
            df = pd.DataFrame(data=torch.permute(preds[i], (1,0)), columns=['Predicted_EW', 'Predicted_NS'])
            df['TimeIndex'] = df.index
            df['ObjectID'] = splid.get_id(idx*10 + i)
            df['Predicted_NS'] = splid.le_type.inverse_transform(df['Predicted_NS'])
            df['Predicted_EW'] = splid.le_type.inverse_transform(df['Predicted_EW'])
            frames.append(utils.convert_classifier_output(df))

df_preds = pd.concat(frames)

evaluator = evaluation.NodeDetectionEvaluator(pd.read_csv(ground_truth), df_preds,
                                              tolerance=config.tolerance)

precision, recall, f2, rmse = evaluator.score()
print(f'Precision for the full set: {precision:.2f}')
print(f'Recall for the full set: {recall:.2f}')
print(f'F2 for the full set: {f2:.2f}')
print(f'RMSE for the full set: {rmse:.2f}')

# Plot the evaluation timeline for a random ObjectID from the training set
evaluator.plot(np.random.choice(df_preds['ObjectID'].unique()))

# Loop over the Object IDs in the training set and call the evaluation
# function for each object and aggregate the results
total_tp = 0
total_fp = 0
total_fn = 0
for oid in df_preds['ObjectID'].unique():
    tp, fp, fn, gt_object, p_object = evaluator.evaluate(oid)
    total_tp += tp
    total_fp += fp
    total_fn += fn

print(f'Total true positives: {total_tp}')
print(f'Total false positives: {total_fp}')
print(f'Total false negatives: {total_fn}')