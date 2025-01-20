This repository contains the .py CLI version of the code for the paper: Efficient Hierarchical Contrastive Self-supervising Learning for Time Series Classification via Importance-aware Resolution Selection
This paper was Accepted to the IEEE Big Data 2024 conference. 

# Abstract
Recently, there has been a significant advancement in designing Self-Supervised Learning (SSL) frameworks for time series data to reduce the dependency on data labels. Among these works, hierarchical contrastive learning-based SSL frameworks, which learn representations by contrasting data embeddings at multiple resolutions, have gained considerable attention. Due to their ability to gather more information, they exhibit better generalization in various downstream tasks. However, when the time series data length is significant long, the computational cost is often significantly higher than that of other SSL frameworks. In this paper, to address this challenge, we propose an efficient way to train hierarchical contrastive learning models. Inspired by the fact that each resolution's data embedding is highly dependent, we introduce importance-aware resolution selection based training framework to reduce the computational cost. In the experiment, we demonstrate that the proposed method significantly improves training time while preserving the original model's integrity in extensive time series classification performance evaluations.

<div align=center>
  <img src=new_fig1.jpg>
</div>

# Important for running:
This repo works via CLI and can be ran by following these steps: 

Firstly for each respective Dataset you need to create a corresponding folder with the same name in the IARS/figures directory in order to save the figures to the corresponding dataset.

ex. IARS/Datasets/**Example_Dataset_Name** and IARS/figures/**Example_Dataset_Name** both must exist before running the script.
# Requirements:
```bash
  git clone https://github.com/KEEBVIN/IARS.git
  pip install -r requirements.txt
```

# Available Datasets
There are 7 available datasets that're ready to use once the repo and requirements are installed. Due to file sizing constraints we could not include the larger datasets that were used in testing in the paper.
- AtrialFibrilation
- Cricket
- EthanolConcentration
- SelfRegulationSCP1
- SelfRegulationSCP2
- StandWalkJump

# Available Arguments
- **lr** : Learning Rate
- **e** : Number of training Epochs
- **i** : Number of training loops
- **ds** : Used Dataset
- **dir** : Change the figure directory location
- **h** : Help screen
# Sample Script
```bash
  python run.py --lr 0.001 --e 100 --i 5 --ds StandWalkJump 
```
# Default Configuration
```bash
python run.py
```

