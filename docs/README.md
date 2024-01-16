<div align="center">

# SPLID
**Satellite Pattern-of-Life Identification Dataset.**

______________________________________________________________________

<p align="center">
  <a href="https://splid-devkit.readthedocs.io/en/latest/installation.html">Installation</a> •
  <a href="https://splid-devkit.readthedocs.io/en/latest/dataset.html">SPLID Dataset</a> •
  <a href="https://github.com/ARCLab-MIT/splid-devkit">Development Kit</a> •
  <a href="https://splid-devkit.readthedocs.io/en/latest/metric.html">Metric</a> <br>
  <a href="https://www.researchgate.net/publication/374083350_AI_SSA_Challenge_Problem_Satellite_Pattern-of-Life_Characterization_Dataset_and_Benchmark_Suite">Paper</a> •
  <a href="https://splid-devkit.readthedocs.io/en/latest/resources.html">Resources</a> •
  <a href="https://splid-devkit.readthedocs.io/en/latest/README.html">Documentation</a> •
  <a href="https://eval.ai/web/challenges/challenge-page/2164/overview">Challenge</a>
  
</p>

[![Python](https://img.shields.io/badge/python-%20%203.11-blue.svg)]()
[![Documentation Status](https://readthedocs.org/projects/splid-devkit/badge/?version=latest)](https://splid-devkit.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

______________________________________________________________________
<div align="left">
  
## Getting started
The Satellite Pattern-of-Life Identification Dataset (SPLID) can be downloaded from <a href="https://www.dropbox.com/scl/fo/jt5h1f82iycjb8elybmlz/h?rlkey=bjcmny486ddf7m0j7b9uok9ww&dl=0">here</a>. You can find installation instructions for the development toolkit (devkit) on the [installation page](https://splid-devkit.readthedocs.io/en/latest/installation.html), and detailed information about the dataset is available on the [dataset page](https://splid-devkit.readthedocs.io/en/latest/dataset.html).

For those interested in participating in the satellite pattern-of-life identification challenge, please refer to the competition [landing page](https://eval.ai/web/challenges/challenge-page/2164/overview) for complete details.

To get acquainted with the SPLID development toolkit, we recommend following these steps:
- <b>Read the SPLID Paper:</b> Start by reading the paper on the [Satellite Pattern-of-Life Characterization Dataset and Benchmark Suite](https://www.researchgate.net/publication/374083350_AI_SSA_Challenge_Problem_Satellite_Pattern-of-Life_Characterization_Dataset_and_Benchmark_Suite).
- <b>Explore Dataset Descriptions:</b> Review comprehensive descriptions of the [SPLID dataset](https://splid-devkit.readthedocs.io/en/latest/dataset.html) to understand its nuances.
- <b>Setup Toolkit and Dataset:</b> Follow the step-by-step instructions provided on the [installation page](https://splid-devkit.readthedocs.io/en/latest/installation.html) to set up the devkit and dataset.
- <b>Tutorials to Get Started:</b> We've prepared tutorials in the [baseline_submissions folder](https://github.com/ARCLab-MIT/splid-devkit/tree/main/baseline_submissions), each covering essential topics:
    - `data_visualization.ipynb`: This tutorial guides you on loading, manipulating, and visualizing the training data and labels.
    - `heuristic_baseline.ipynb`: Here, you'll find a tutorial demonstrating the baseline heuristic solution. It explains the heuristic approach, illustrates how performance is measured, and provides visualizations of the results.
    - `ml_baseline.ipynb`: This tutorial takes you through the development of the baseline machine learning solution. It describes the training workflow, showcases how performance is assessed, and offers visualizations of the results.

______________________________________________________________________

## Background
Because most longitudinal positions within the geostationary belt are unstable—small perturbative forces in the near-Earth space environment push and pull objects from their stations—operators regularly command their geosynchronous (GEO) satellites to perform station-keeping maneuvers to maintain a near-fixed position relative to the Earth’s surface. The magnitude, direction, and frequency of station-keeping maneuvers, however, vary alongside GEO satellites’ mission requirements, the constraints of their onboard control systems, and other factors, including their operators’ preferences. When satellite operators choose to change the position of their GEO satellites by performing a longitudinal-shift maneuver, a once rare, but now common occurrence in that orbital regime, they can choose to perform it as they please, with wide observable variations in drift rate, drift time, and Δv expended. Similarly, operators retire their satellites at their own discretion, sometimes leaving them in libration orbits—where they slowly swing past one or both of the geostationary belt’s stable points—but sometimes placing them in higher-altitude graveyard orbits. Operator decisions that affect GEO satellites’ physical positions leave traces in the satellites' historical orbital elements. A GEO satellite’s pattern of life (PoL) summarizes these traces by offering a description of satellites’ unique combinations of behavioral modes—time periods in which they were station-keeping or drifting—separated by nodes during which they changed from one mode to another.
(Adapted from [Geosynchronous Satellite Behavior Classification via Unsupervised Machine Learning, by Thomas G. Roberts, Haley E. Solera, and Richard Linares.](https://www.researchgate.net/publication/368982563_Geosynchronous_Satellite_Behavior_Classification_via_Unsupervised_Machine_Learning))

______________________________________________________________________

## Citation
Please use the following citation when referencing the dataset and [challenge problem](https://www.researchgate.net/publication/374083350_AI_SSA_Challenge_Problem_Satellite_Pattern-of-Life_Characterization_Dataset_and_Benchmark_Suite):
```
@inproceedings{splid, 
  title={AI SSA challenge problem: satellite pattern-of-life characterization dataset and benchmark suite},
  author={Siew, Peng Mun and Solera, Haley E and Roberts, Thomas G and Jang, Daniel and Rodriguez-Fernandez, Victor and How, Jonathan P and Linares, Richard},
  booktitle={Proceedings of the 2023 Advanced Maui Optical and Space Surveillance Technologies Conference (AMOS), Wailea Beach Resort, Maui, HI, USA},
  year=2023
}
```
______________________________________________________________________

## Contact us
Submit an issue to the github repo or email us at ai_challenge@mit.edu.

## Acknowledgements
We'd like to extend our sincere gratitude to EvalAI for hosting this challenge and providing a robust platform. Special thanks to the team behind the [My Seizure Gauge Forecasting Challenge 2022](https://github.com/seermedical/msg-2022), as their detailed documentation was invaluable for the successful setup of our own competition.
