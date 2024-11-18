This gihub repository obtains the original jupiter notebook file used for the research paper titled: Efficient Hierarchical Contrastive Self-supervising Learning for Time Series Classification via Importance-aware Resolution Selection
This paper was submitted to the IEEE Big Data 2024 conference

# Abstract
Recently, there has been a significant advancement in designing Self-Supervised Learning (SSL) frameworks for time series data to reduce the dependency on data labels. Among these works, hierarchical contrastive learning-based SSL frameworks, which learn representations by contrasting data embeddings at multiple resolutions, have gained considerable attention. Due to their ability to gather more information, they exhibit better generalization in various downstream tasks. However, when the time series data length is significant long, the computational cost is often significantly higher than that of other SSL frameworks. In this paper, to address this challenge, we propose an efficient way to train hierarchical contrastive learning models. Inspired by the fact that each resolution's data embedding is highly dependent, we introduce importance-aware resolution selection based training framework to reduce the computational cost. In the experiment, we demonstrate that the proposed method significantly improves training time while preserving the original model's integrity in extensive time series classification performance evaluations.

# Important for running:
Since our codebase resides in google-colab we use google drive to store and retrieve our data, figures, and model. This means that before you can run the notebook you must firstly install the respective datasets, assign the correct paths for saving/loading the figures, model, and data. 

These portions of the data should be changed respectively: 

The dataset paths: 

![dataset paths](https://github.com/KEEBVIN/IARS/blob/main/readme_images/datasets.png)

Training data and labels path:

![dataset paths](https://github.com/KEEBVIN/IARS/blob/main/readme_images/data_paths.png)

Plot saving path:

![dataset paths](https://github.com/KEEBVIN/IARS/blob/main/readme_images/plot_saving.png)

These are a few path examples where changes must be made before the code is able to completely run. 

