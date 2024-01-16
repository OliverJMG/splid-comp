# Usage
In this section, usage of the development toolkit is demonstrated via guided examples.

## Directory Structure
```
.
├── dataset                  
│   ├── train                # Contains the training dataset
│   │   ├── 1.csv            # Time-series data for object with ID 1
│   │   ├── 2.csv            # Time-series data for object with ID 2
│   │   └── ...
├── labels.csv               # Contains the annotations for the train dataset
└── baseline_submissions
    ├── heuristic_python     # Contains the heuristic baseline
    └── ml_python            # Contains the machine learning baseline

```
We will be using JupyterLab for these examples. First navigate to the cloned GitHub `splid-devkit` folder from the terminal and run the following command to launch JupyterLab.
```
jupyter lab
```
JupyterLab will then open automatically in your browser. The example submissions are located in the `baseline_submissions` folder.

## Data Visualization
The data visualization notebook (`data_visualization.ipynb`) guides the challengers in loading, manipulating, and visualizing the training data and labels.

## Heuristic-based Baseline
The following subsection guides the challengers in implementing a heuristic-based model. Readers are referred to the [baseline solutions page](https://splid-devkit.readthedocs.io/en/latest/baseline.html#heuristic-based-approach) for more information about this approach.

To be updated.

## Machine Learning-based Baseline
The following subsection guides the challengers in implementing a Random Forest classifier model. Readers are referred to the [baseline solutions page](https://splid-devkit.readthedocs.io/en/latest/baseline.html#machine-learning-based-approach) for more infomation of this approach.


## Performance Evaluation
The `evaluation.py` script provides a standard way to assess the performance of the models submitted for the challenge. It employs metrics and evaluation techniques that are aligned with the challenge's objectives.

The toy datasets `ground_truth_toy.csv` and `participant_toy.csv` serve as simplified, example datasets for the challenge. These datasets are intended for initial testing and understanding of the evaluation script and the baseline model.

**Example Usage**
The `run_evaluator` function is the main entry point of the script and accepts the following arguments:

- `participant`: Path to the participant's CSV file.
- `ground_truth`: Path to the ground truth CSV file.
- `plot_object`: Object ID for which to plot evaluation details.

You can also run the script directly from the command line. For example:
```bash
python evaluation.py --participant=participant.csv --ground_truth=ground_truth.csv --plot_object=12345
```
This example assumes you have `participant.csv` and `ground_truth.csv` files in the expected directories, and you want to plot evaluation details for object ID `12345`. If no arguments are provided, the evaluation will be run for the toy datasets.

**Returns**
The `score` function within the file returns the evaluation metrics as per the challenge guidelines (that is, the F<sub>2</sub> and the RMSE). Additionally, the precision and recall are also returned, and, if the `plot_object` parameter is provided, it generates plots for that specific object ID to aid in understanding the evaluation.
