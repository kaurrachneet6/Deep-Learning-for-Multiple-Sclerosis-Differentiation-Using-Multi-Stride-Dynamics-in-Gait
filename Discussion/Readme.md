#### Discussion analysis
* **Ablation_TaskGen_MLTraditional.ipynb**: Ablation Study on Task generalization framework W -> WT with Traditional ML models only.
For ablation study, We explore the performance using Spatial, Temporal, Kinetic, Spatiotemporal, Spatial+Kinetic and Temporal+Kinetic subgroups within our all 21 features. 

* **Ablation_SubjectGen_MLTraditional.ipynb**: Ablation Study on Subject generalization framework with Traditional ML models only.

* **Ablation Study for the DL models** is done using changes done to the task/subject generalization utility functions in utils_lstm.py respectively.

* **Permutation Importance based feature importance for the task generalization framework** is done by adding relevant utility functions in utils_lstm.py. Basically, for each feature of interest, say, cadence, we randomly shuffle the cadence feature in our data 5 times and each time compute the evaluation metrics by predicting on this new shuffled test set data. The trained model used for predictions is the best tuned model for the task generalization framework. 

* **Permutation Importance based feature importance for the subject generalization framework** is done by adding relevant utility functions in utils_lstm.py. Basically, for each feature of interest, say, cadence, we randomly shuffle cadence feature in our data 5 times and each time compute the evaluation metrics by predicting on this new shuffled test set data for each of the 5 test folds in the 5-fold CV. The training folds remain as is and the best tuned subject gen model is used for making predictions on the shuffled test folds.

* **PermutationImportance_Vizualizations.ipynb**: Vizualizing the permutation feature importance results for best task and subject generalization models.

* **TaskGen_LowDimVizualizations.ipynb**: Extracting and low dimensional vizualizations for the last layer features from the best task generalization model (MSResnet). This analysis is only done on regress-N data and best task gen model (for the regress-N data). 

* **SubjectGen_LowDimVizualizations.ipynb**: Extracting and low dimensional vizualizations for the last layer features from the best subject generalization model

* **SubjectGen_SeverityVizualization.ipynb**: Vizualizing the predictions for best subject generalization model with the MS severity and SPPB lower extremity strength total scores 

* **RawTreadmillDataExtraction.ipnb**: Extracting raw treadmill features (COPX, COPY, ForceZ and belt speed) for each stride and time normalizing data in a given stride, and so retaining 30 samples per stride after downsampling with smooting approach.
* **SizeNScalingRawTreadmillData.ipynb**: Size-N normalizing raw treadmill features (COPX, COPY, ForceZ and belt speed) for each stride. COP should be normalized by length (height), speed by (leg length or similar as gait speed), and force by body weight.
