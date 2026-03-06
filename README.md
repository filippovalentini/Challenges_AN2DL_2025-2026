# Overview
This repository contains the work related to the two Kaggle challenges performed in the context of the 2025/26 Artificial Neural Networks and Deep Learning course at Politecnico di Milano. 
Both challenges were graded with a 5/5 score.

## Team members
- Marcello Braghiroli
- Guido Cestele
- Filippo Valentini

# First Challenge
![First Challenge Image](challenge1.png)

## Dataset
The "Pirate Pain Dataset" is based on multivariate time series data, captured from both ordinary folk and pirates over repeated observations in time. Each sample collects temporal dynamics of body joints and pain perception, with the goal of predicting the subject’s true pain status:
- *no_pain*
- *low_pain*
- *high_pain*

Each record represents a time step within a subject’s recording, identified by sample_index and time. The dataset includes several groups of features:
- *pain_survey_1–pain_survey_4* -- simple rule-based sensor aggregations estimating perceived pain.
- *n_legs, n_hands, n_eyes* - subject characteristics.
- *joint_00–joint_30* -- continuous measurements of body joint angles (neck, elbow, knee, etc.) across time.

We were provided both with a labeled training set and an unlabeled test set; the latter was used to evaluate the performance of our network and to determine the position of our team on the Kaggle leaderboard.

## Goal
Develop a Recurrent Neural Network to predict the real pain level of each test subject based on their time-series motion data.

## Results

| Experiment | Test F1 Score |
|:------- |:----------- |
| Simple LSTM NTW | 0.9111 |
| BiGRU NTW + Oversampling  | 0.9461 |
| **BiGRU NTW + FC layer + Adaptive Stride** | **0.9563** |
| BiGRU NTW + FC layer + Convolutional layer | 0.9418 |

# Second Challenge
![Second Challenge Image](challenge2.png)

## Dataset
The provided dataset contains several diseased human tissue images of different sizes, each combined with a binary mask (identifying the regions most likely to contain the diseased tissue) and a label describing the molecular subtype of the disease:
- *Luminal A*
- *Luminal B*
- *HER2(+)*
- *Triple Negative*

## Goal
Develop a Convolutional Neural Network to predict the correct molecular subtype of each test image based on their microscopic tissue morphology.

## Results

| Experiment | Test F1 Score |
|:------- |:----------- |
| EfficientNetB3 | 0.3854 |
| ConvNeXt  | 0.4281 |
| Ensembling of 3 pretrained FENs | 0.3929 |
| ConvNeXt + multiscaling | 0.3709 |
| **RetCCL ResNet-50** | **0.4407** |
| Prov-GigaPath | 0.4279 |
