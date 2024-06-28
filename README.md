# Bridging Fairness and Privacy: An Experimental Analysis of Bias Assessment in Federated Learning
This repository contains the work from "A Holistic Analysis of Bias Assessment in Federated Learning", a master thesis by Jelke Matthijsse. This work assesses group biases within privacy-preserving federated learning. The focus of this research is two-fold. First, this research assesses the correctness of the currently used aggregated local bias detection approaches in federated learning. Secondly, this research assesses biases that can be perpetuated by the federated learning framework. 

This evaluation is done by comparing three model pipelines. These pipelines are compared with each other, according to Figure 4.1. 
1. Aggregated local bias assessment of a global model trained with federated learning.
2. Global bias assessment of a global model trained with federated learning.
3. Central bias assesment of a central model trained with central learning.

![image](https://github.com/jelkejm/thesis/assets/77006994/26a1bb6c-310d-4e9a-ac33-aec7abac4ee9)



This README provides all information necessary to reproduce the work as proposed in "A Holistic Analysis of Bias Assessment in Federated Learning". This README first provides an overview of the repository. Secondly, instructions on the requirements are given. Then it will be explained how the necessary data can be obtained. Thereafter, an explenation of the experimental setup is provided. Lastly, it will be discussed how the results from the experiments can be interpreted.

# Directory structure
```
├── acs_data
|    └── all .csv files
├── experiments/
|    └── all 17 experiments_{..}.py files
├── results/
|    └── all 17 results_experiments_{..}.txt files
|    └── results.ipynb
├── download_acs_data.ipynb
├── data_analysis_acs.ipynb
├── utils.py
├── fl_pipeline.py
├── train_test.py
├── data_loader.py
└── requirements.txt
```

# Environment and Requirements
This repository includes a `requirements.txt` file with all necessary `pip` packages that should be installed. A virtual environment can be created with an environment manager of choice (e.g. conda) where the requirements can be installed with:

```
# Create env
conda create -n fl_bias python=3.11 && conda activate fl_bias
# Install requirements
pip install -r requirements.txt
```
# Data
All experiments are conducted using the ACS Income dataset, a subset of the ACS PUMS dataset. The data can be downloaded by running `download_acs_data.ipynb`. All experiments are done using the `person` survey, over a `1-year` coverage for the year `2022`. Make sure that all data files, for every US state, are put in the `acs_data` folder.  

# Experiments
The experimental setup of this research consist of a comparison between three pipelines considering multiple experimental data partitions that consider varying types of data heterogeneity between federated learning clients. Every `experiments_{}.py` file trains and evaluates all three model pipelines, given their corresponding experimental data partition.

Every experimental file can be run with 

```
python your_script.py
    --num_clients 51
    --learning_rate 0.001
    --num_rounds 5
    --epochs 5
    --fraction_fit 1.0
    --fraction_evaluate 1.0
    --perc 0.75 
    --seeds 42 123 456
    --result_path 'new_results.txt'
```

The hyperparameters that can be adjusted within each experimental setup are:
- `num_clients`: the number of clients used for the data partition
- `learrning_rate`: the learning rate across for local and global model training
- `num_rounds`: the number of rounds that is used in federated learning
- `epochs`: the number of epochs used for <ins>central</ins> training
- `fraction_fit`: percentage of available clients that will be sampled for training in FL
- `fraction_evaluate`: percentage of available clients that will be sampled for evalutation in FL
- `perc`: percentage for heterogeneity categorization (i.e. the split for every client, e.g., 75-25, 85-15)
- `seeds`: the different seeds over which will be run
- `result_path`: the path to which the results will be written

<img src="https://github.com/jelkejm/thesis/assets/77006994/e7a25b91-6890-4be9-ae36-1b0d54df4644" width="800" alt="image description">

# Results
All figures from the original paper can be obtained by running `results.ipynb`, given the experimental result files in the `/results` folder.
