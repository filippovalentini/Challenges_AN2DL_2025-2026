# Overview
This repository contains the work performed during two Kaggle challenges in the context of the 2025/26 Artificial Neural Networks and Deep Learning course at Politecnico di Milano.

# First Challenge
![First Challenge Image](challenge1.png)

## Dataset
The "Pirate Pain Dataset" is based on multivariate time series data, captured from both ordinary folk and pirates over repeated observations in time. Each sample collects temporal dynamics of body joints and pain perception, with the goal of predicting the subject’s true pain status:
- no_pain
- low_pain
- high_pain

Each record represents a time step within a subject’s recording, identified by sample_index and time. The dataset includes several groups of features:
- pain_survey_1–pain_survey_4 -- simple rule-based sensor aggregations estimating perceived pain.
- n_legs, n_hands, n_eyes —- subject characteristics.
- joint_00–joint_30 —- continuous measurements of body joint angles (neck, elbow, knee, etc.) across time.

For the challenge, we were provided both with a labeled training set and an unlabeled test set. The latter was used to evaluate the performance of our network and to determine the position on the Kaggle leaderboard.
