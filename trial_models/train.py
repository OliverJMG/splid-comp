import os
import pickle
import tempfile
from functools import partial

import torch
from fastcore.basics import Path, AttrDict
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from sklearn.model_selection import train_test_split
from torch.utils.data import random_split

from dataset import SPLID
from loss import WeightedCELoss
from models.prectime import PrecTime


def load_data(data, gnd):
    train_data, test_data = train_test_split(data, test_size=0.15, random_state=42)
    cols = ['Inclination (deg)', 'Longitude (deg)', 'Eccentricity', 'Semimajor Axis (m)', 'RAAN (deg)',
            'Argument of Periapsis (deg)', 'Vz (m/s)']
    classes = ['ES-ES', 'SS-CK', 'SS-EK', 'SS-HK', 'SS-NK', 'IK-CK', 'IK-EK', 'IK-HK', 'ID-NK', 'AD-NK']

    trn_data = SPLID(train_data, gnd, cols, classes=classes)
    tst_data = SPLID(test_data, gnd, cols, classes=classes)

    return trn_data, tst_data


def train_model(config, data, gnd):
    n_win = config["n_win"]
    l_win = 2208 // n_win
    model = PrecTime(10, n_win=n_win, l_win=l_win, c_in=7, c_conv=config["c_conv"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    model.to(device)

    class_weights = torch.tensor([0.1, 0.2, 0.2, 0.2, 0.2, 4, 4, 4, 4, 4]).cuda()
    criterion = WeightedCELoss(class_weights=class_weights)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            opt.load_state_dict(optimizer_state)

    trainset, _ = load_data(data, gnd)

    test_abs = int(len(trainset) * 0.85)
    train_subset, val_subset = random_split(
        trainset, [test_abs, len(trainset) - test_abs]
    )

    trainloader = torch.utils.data.DataLoader(
        train_subset, batch_size=4, shuffle=True, num_workers=4
    )
    valloader = torch.utils.data.DataLoader(
        val_subset, batch_size=4, shuffle=True, num_workers=4
    )

    for epoch in range(20):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_steps = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            fine_out, coarse_out = model(inputs)
            tot_loss, fine_loss, coarse_loss = criterion(fine_out, coarse_out, labels)
            tot_loss.backward()
            opt.step()

            # print statistics
            running_loss += tot_loss.item()
            epoch_steps += 1
            if i % 200 == 199:  # print every 2000 mini-batches
                print(
                    "[%d, %5d] loss: %.3f"
                    % (epoch + 1, i + 1, running_loss / epoch_steps)
                )
                running_loss = 0.0

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                fine_out, coarse_out = model(inputs)
                _, predicted = torch.max(fine_out.data, 1)
                total += labels.size(0) * labels.size(1) * labels.size(2)
                correct += (predicted == labels).sum().item()

                tot_loss, fine_loss, coarse_loss = criterion(fine_out, coarse_out, labels)
                val_loss += tot_loss.cpu().numpy()
                val_steps += 1

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), opt.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": (val_loss / val_steps), "accuracy": correct / total},
                checkpoint=checkpoint,
            )
    print("Finished Training")


def test_accuracy(net, data, gnd, device="cpu"):
    trainset, testset = load_data(data, gnd)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            fine_out, coarse_out = net(images)
            _, predicted = torch.max(fine_out.data, 1)
            total += labels.size(0) * labels.size(1) * labels.size(2)
            correct += (predicted == labels).sum().item()

    return correct / total


def main(num_samples=12, max_num_epochs=20, gpus_per_trial=1):
    conf = AttrDict(
        challenge_data_dir=Path('~/Projects/splid-comp/dataset').expanduser(),
    )
    train_data_dir = conf.challenge_data_dir / "train_v2"
    ground_truth = conf.challenge_data_dir / 'train_labels_v2.csv'

    datalist = []

    # Searching for training data within the dataset folder
    for file in os.listdir(train_data_dir):
        if file.endswith(".csv"):
            datalist.append(os.path.join(train_data_dir, file))

    # Sort the training data and labels
    datalist = sorted(datalist, key=lambda i: int(os.path.splitext(os.path.basename(i))[0]))

    search_space = {
        "n_win": tune.choice([46, 92, 138, 184]),
        "c_conv": tune.choice([128, 256]),
    }

    scheduler = ASHAScheduler(
        max_t=max_num_epochs,
        grace_period=10
    )

    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model, data=datalist, gnd=ground_truth),
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=search_space,
    )

    results = tuner.fit()

    best_trial = results.get_best_result("loss", "min")

    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.metrics['loss']}")
    print(f"Best trial final validation accuracy: {best_trial.metrics['accuracy']}")

    n_win = best_trial.config["n_win"]
    l_win = 2208 // n_win

    best_trained_model = PrecTime(10, n_win=n_win, l_win=l_win, c_in=7,
                                  c_conv=best_trial.config["c_conv"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
    best_trained_model.to(device)

    checkpoint_path = os.path.join(best_trial.checkpoint.to_directory(), "checkpoint.pt")
    model_state, optimizer_state = torch.load(checkpoint_path)
    best_trained_model.load_state_dict(model_state)

    test_acc = test_accuracy(best_trained_model, datalist, ground_truth, device)
    print("Best trial test set accuracy: {}".format(test_acc))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=12, max_num_epochs=20, gpus_per_trial=1)
