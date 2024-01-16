# Submission Tutorial
Participants will be submitting their trained algorithm with their  code/workflow to generation predictions into the competition platform, in the programming language they prefer. Here we provide Docker containers and example submissions for Python, but participants are allowed to submit with other custom containers too. The performance of their trained AI algorithm will then be evaluated against our internal test data and evaluation metric on our remote server. Once the evaluation is complete, the results are sent back to the live leaderboard on the challenge platform. This allows for the usage of a private test set, which helps with data privacy concerns and prevents data leakage.

## Prepare code for the submission
### Input data for docker container
The predictions script **must** load data from the `/dataset/test`directory in the docker container.

> Note: `/dataset` is an absolute directory path, mounted on the root directory of the container. It is not a path relative to the current working directory.

The data, as for the training set, is stored in our remote server as separate files with the following directory structure:
```
/
├── dataset                  
    ├── test                
        ├── test1.csv            # Ephems for object with ID test1
        ├── test2.csv            # Ephems for object with ID test2
        └── ...
```

### Required outputs
Your submission has to generate predictions, and save them as a `.csv` file.

- The csv file should be saved to `/submission/submission.csv` in the docker container. Note that `/submission` is an absolute directory path, mounted on the root directory of the container. It is not a path relative to the current working directory.
- The output `csv` file shoud have four columns:
    - `ObjectID`
    - `TimeIndex`: Relative to the first observation of the object, i.e., row number in the ephems table. Starts in `0`.
    - `Direction`: `EW` (East-West) or `NS` (North-South)
    - `Node`: Node label. See more in the [dataset page](https://splid-devkit.readthedocs.io/en/latest/dataset.html)
    - `Type`: Node type. See more in the [dataset page](https://splid-devkit.readthedocs.io/en/latest/dataset.html)

Example of output `csv`:
```
ObjectID,TimeIndex,Direction,Node,Type
1,0,EW,SS,CK
1,75,EW,ID,NK
1,0,NS,SS,HK
1,115,NS,IK,EK
2,25,EW,SK,CK
2,45,NS,IK,EK
```

### Computational restrictions for submissions
#### Hardware
The submission you make must take the following computational resource limits into consideration and not exceed them.

- **CPU** : 4 cpu cores will be available to you.
- **RAM** : Approx 16Gb of memory will be available.
- **GPU** : 1 GPU, with 16Gb of memory will be available.
- **Run Time** : Your submission should run within approx 10 minutes or less. If submissions take too long to run, they will fail.

#### Software
For maximum compatibility in Python-based submissions, it is recomended to use `python3.10``, and pin your python libraries to the following suggested versions (but only if you actually need those libraries).
```
scikit-learn==1.3.2
```

### Test the submission locally
It is strongly recomended that you test run your docker container before making a submission. This will allow you catch any potential bugs you may have in your code.

To test that your Docker container does the right thing, do the following:

#### 1. Prepare dummy version of the test dataset
- Download **[TODO: this zip file]** containing small test data (the data used in the warmup phase of the challenge).
- Extract it somewhere. We will refer to this path as `LOCAL_DATA_DIR`.
- You should end up with a path that looks like `LOCAL_DATA_DIR/test/`, and directory structure like:
```
LOCAL_DATA_DIR
    test
        O1.csv
        O2.csv
        O3.csv
        ...
```

#### 2. Run the container
Open a terminal located in the root folder of your submission files (where your `Dockerfile` should be), and run:
```
docker run --rm\
    --name mysubmission\
    -v ${LOCAL_DATA_DIR}:/dataset\
    -v ${LOCAL_PREDICTIONS_DIR}:/submission\
    splid-submission
```

In this command, `LOCAL_DATA_DIR` points to the path where you have extracted the test data, and `LOCAL_PREDICTIONS_DIR` points to the place you want to see the results once the docker container finishes. This command will:
- Mount your local data dir to `/dataset` on the container.
- Mount your local predictions dir to `/submission` on the container.
- Run the entrypoint script you specified in your docker container (e.g. `python -m submission.py` for the [sample Dockerfile](../baseline_submissions/ml_python/Dockerfile) for the Python-based ML baseline).

> Note: You can replace the `splid-submission` with whatever name and tag you used to build the container.

> Note 2: To test GPU, you will need to install the [nvidia container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html): You will also need to use `--gpus all` in the call to `docker run`.

#### 3. Check the outputs
If the above call to `docker run` finished correctly with your submission code, there should be a `LOCAL_PREDICTION_DIR/TEST_PREDICTIONS_FILENAME.csv`, where `TEST_PREDICTIONS_FILENAME` refers to the filename that you have used in your submission script to store your results (e.g., `test_predictions` in the provided ML baseline submission code). Open that file and ensure it is structured correctly. Once you do it, you're ready to submit your submission code to our servers!

## Upload the submission to EvalAI

### Prepare the environment for uploading submissions
1. Sign up for [the competition on Eval AI](XXX).
2. Install evalai cli tool
    ```bash
    pip install evalai
    ```

3. Set your EvalAI authentication token on the cli tool. You can get the token by going to your [eval.ai profile](https://eval.ai/web/profile)
    ```bash
    evalai set_token XXXXXXXXXXXX
    ```

### Push the submission

Open a terminal and run:

```bash
# WARM-UP PHASE
# evalai push MY_DOCKER_IMAGE:MY_TAG --phase warmup-2164
evalai push splid-submission:latest --phase mit-warmup-2164

# COMPETITION PHASE
# evalai push MY_DOCKER_IMAGE:MY_TAG --phase competition-2164
evalai push splid-submission:latest --phase mit-competition-2164
```

> Note: Make sure you substitute `splid-submission:latest` with the actual docker image name and tag you built.

This will upload your docker image to our remote servers, and trigger an evaluation. Note, the first time you submit, it might take a while to upload. Subsequent uploads will be quicker if the base layers of your docker container are the same as the previous submission.

You can now monitor the progress of the submission in the [my submissions](https://eval.ai/web/challenges/challenge-page/2164/my-submission) section in the competition dashboard.


## [Optional] Create your own Docker image for the submission
If you want to create your own Docker image for your submission instead of using the provided [Python](../baseline_submissions/ml_python/Dockerfile) submissions (for example, because you're using a different language), here are some considerations to take into account.

### GPU compatibility
If you want to make use of the GPU resources that are on the cloud infrastructure, then you should use a docker image that contains Nvidia cuda, and cudnn drivers. for example, `nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04`. 

> Note: To use GPU in your container locally, you will need to install the [nvidia container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html). You will also need to use --gpus all flag with docker.

### Tips
1. Use `python -m pip install` instead of `pip install` for installing python libraries.

2. Set the `--no-cache-dir` argument when installing python libraries. This keeps docker images smaller.

    ```bash
    # Example
    python -m pip install --no-cache-dir pandas
    ```

3. After installing linux packages, delete the repository cache (this keeps docker images smaller)

    ```bash
    # Exammple of code to delete the apt package cache
    rm -rf /var/lib/apt/lists/*
    ```

4. Only install the bare minimum you need to run predictions. It will keep docker images smaller.
    - During the early stages, you will probably make use of a lot of libraries to perform exploratory analysis. E.g., for visualizing data, creating plots, and exporting images.
    - Do not install all of those libraries on your system onto the Docker image that you submit. 
        - E.g. there is probably no need to be installing `matplotlib` or `plotly` to make predictions.
