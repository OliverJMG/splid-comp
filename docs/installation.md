# Installation
Here, the step-by-step instructions to install the devkit is provided.

## Install Docker

### 1. Install prerequisites

- Windows: WSL2. To install WSL2:
    1. Open PowerShell as Administrator and run `wsl --install`.
    2. Restart your machine when prompted.
    3. Install a Linux distro from Microsoft Store (e.g., Ubuntu).
    4. Set WSL 2 as default: `wsl --set-default-version 2`.
    5. Verify WSL version: `wsl --list --verbose`.
- macOS: Homebrew
- Linux: `sudo apt update && sudo apt upgrade`

### 2. Download Docker installer
- Windows: Docker Desktop from official site
- macOS: `brew install --cask docker`
- Linux: `sudo apt install docker.io``

### 3. Run installer
- Windows: Double-click `.exe` file
- macOS: Follow Brew prompts
- Linux: `sudo systemctl start docker && sudo systemctl enable docker`

### 4. Verify installation
- Run `docker --version`
- Run `docker run hello-world``

## [Optional] Install Nvidia container-toolkit
If you plan to use GPU power in your submission and you want to try it locally, you will need to install the [nvidia container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).

## Install Python
If you do not have Python installed, please refer to the anaconda installation instructions [here](https://docs.anaconda.com/free/anaconda/install/index.html). The devkit is tested for Python 3.8+ on Ubuntu and Mac OS.

## Download the Devkit
Download the devkit using the terminal and move into the new folder named "splid-devkit" that was automatically created:
```
cd && git clone https://github.com/ARCLab-MIT/splid-devkit.git && cd splid-devkit
```
The above command will download the files to your home directory. While you can change this to an arbitrary directory, the rest of our tutorials assume that you are using the home directory.

Install the required packages by running the following command:
```
pip install -r requirements.txt
```

## Download the dataset
The challenge dataset can be downloaded from [here](google drive link). Please store the downloaded dataset into the `~/splid-devkit/dataset` folder. Description of the dataset is available on the [SPLID dataset page](https://splid-devkit.readthedocs.io/en/latest/dataset.html).

