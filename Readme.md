Decomposing Spatio-Temporal Heterogeneity: Matrix-Informed Ensemble Learning for Interpretable Prediction

Basic Requirements:
python=3.7.10
torch>=1.9.0

The two datasets are in /data
The output features of base learners are in /baselearner_features
The MI-EL model is in /model

Training:
use the .py file in /Baselearner-master to train and obtain the outputs of the three base learners.
use main_traffic.py and main_pm25.py to train and obtain the final prediction results.

Testing:
The best weights are in /weights which can be used directly for testing.