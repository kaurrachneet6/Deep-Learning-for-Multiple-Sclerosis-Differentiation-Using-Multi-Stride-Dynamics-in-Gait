# Deep learning-based gait data analysis for Multiple Sclerosis classification

### Authors:
* Rachneet Kaur rk4@illinois.edu https://www.linkedin.com/in/rachneet-kaur-a1ba5354/
* Josh Levy jdlevy3@illinois.edu https://www.linkedin.com/in/joshlevymn/
* Manuel Hernandez mhernand@illinois.edu, http://kch.illinois.edu/hernandez
* Richard Sowers r-sowers@illinois.edu, http://publish.illinois.edu/r-sowers/

### Abstract:
**Objective:** Multiple sclerosis (MS) is a chronic neurological condition of the central nervous system leading to various physical, mental and psychiatric complexities. Mobility limitations are amongst the most frequent and early markers of MS. We evaluate the effectiveness of a DeepMS2G (**deep** learning (DL) for **MS** differentiation using **m**ulti-**s**tride dynamics in **g**ait) framework, which is a DL-based methodology to classify multi-stride sequences of persons with MS (PwMS) from healthy older adults (HOA), in order to generalize over newer walking tasks and subjects. 
**Methods:** We collect gait data using an instrumented treadmill from a balanced collection of 20 HOA and 20 PwMS. We utilize domain knowledge-based spatiotemporal and kinetic gait features along with two normalization schemes, namely standard size-based and multiple regression normalization strategies. These normalization help minimize the dependency of extracted gait parameters on demographics. To differentiate between multi-stride sequences of HOA and PwMS, we compare 16 traditional machine learning and DL algorithms.  We observed that residual neural network (ResNet) based models with regression-based normalization were the top performers across both task and subject generality classification designs. Further, we study the interpretability of our best-performing models; and discuss the association between the lower extremity function of participants and our model predictions. 
**Results:** A multi-scale ResNet attained perfect subject classification accuracy and F1-score when generalizing from single-task walking to dual-task walking while talking; and a ResNet resulted in the best subject-wise accuracy and F1 of 82.9% and 0.81, respectively, when generalizing over new participants, both with regression-based normalization. 
**Conclusion:** Our methodological approaches might be a of broader use for problems where multiple observations of dynamic activity are made on necessarily small cohorts (viz. cases of severe illness).
**Significance:** Our proposed DL algorithms may contribute to efforts to automate MS diagnoses. 

### Dependencies:
* Python 3.6
* The versions of Python packages can be found in the file X.txt/X.yml

### Code structure:
#### Data Preparation
* RAW Data:
    * **ButterflyPlot.ipynb**: For plotting the butterfly diagrams, their mean trajectories, and computing the features ((x, y) of the intersection point, ((x-mean_x)^2, (y-mean_y)^2) for the intersection point.
Further, we record the mean intersection point across all strides during the complete walk and standard deviation across interesection points during the entire walk. 
        * The .csv files with features recorded are: ButterflyMeanSD.csv and ButterflyFeatures.csv

    * **FootProgressionAngles.ipnb**: For computing the left and right FPAs and saving to .csv 
        * The .csv file with features recorded is FPA_feature.csv

    * **FeatureExtraction.ipynb**: For creating the final **raw dataframe** combining all the gait features together. 
        * Computes supporting times, stride length, stride width, cadence, stride speed, treadmill speeds, walk ratio, stride time, swing time, stance time, forces. 
        * The .csv files with all the raw features is data/gait_features.csv

* SIZE NORMALIZED Data:
    *  **DS_Scaling.ipynb**: For performing dimensionless scaling based on Hof, At L. "Scaling gait data to body size." Gait & posture 3, no. 4 (1996): 222-223 to the extracted raw gait features.
        * The .csv files with dimensionless scaled features is data/size_normalized_gait_features.csv

* REGRESSION NORMALIZED Data:
    * **RegressFeatureExtraction.ipynb**: For extracting the gait features of the 30 new controls only walking trial dataset for regerssion coeffcient extraction.
    * **MultipleRegressionScaling_controlsTrialW.ipynb**: For preforming multiple regression based scaling of gait features using physical charactersitcs of subjects as independent variables
        * The .csv file with MR scaled features is data/mr_scaled_features.csv 

#### Utility functions 
* **package_imports.py**: package imports
* **utils_traditional_methods.py**: Functions to read extracted features data, genarate sequences for multiple strides, generate summary statistics dataframe, tune and evaluate ML, plot confusion matrices and ROC curves for tuned models in task and subject generalization frameworks.
* **cnn1d_model.py**: CNN1D model for time series classification with and without positional encoding
* **positional_encoding.py**: Positional encoding https://pytorch.org/tutorials/beginner/transformer_tutorial.html 
* **RESNET_model.py**: Residual 1D model for time series classification with and without positional encoding
* **padding.py**: Implementation for "padding = same" in Pytorch https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/padding.py#L28
* **MULTISCALE_RESNET_model.py**: Multiscale Residual network for time series classification https://github.com/geekfeiw/Multi-Scale-1D-ResNet
* **TCN_model.py**: Temporal Convolutional Model 
* **RNN_model.py**: Vanilla Recurrent Neural Network (Uni- and Bi-directional versions)
* **GRU_model.py**: Gated Recurrent Unit model (Uni- and Bi-directional)
* **LSTM_model.py**: Long-short term memory model (Uni- and Bi-directional)
* **CNN_LSTM_model.py**:
* **utils_lstm.py**: Contains definition of general utilities like setting random seed for replicability etc. used across all three generalization frameworks and deep learning models. Further, contains utility functions like train, resume train, evaluate etc. for training the deep learning models on both task and subject generalization frameworks

#### Machine Learning 
* **generate_summary_stats**: This code generates summary statistics (namely mean and standard deviation) over multiple strides of raw, size-normalized and regression-normalized gait features. From 21 gait features across 10 strides, we create a set of 42 mean and standard deviation parameters for each window. The files for raw, size-N and regress-N data are saved as summary_statistics_raw_data.csv, summary_statistics_sizeN_data.csv and summary_statistics_regressN_data.csv respectively.

* **TaskGeneralize_MLTraditional.ipynb**: Traditional ML algorithms on task generalization framework, i.e. train on walking (W) and test on walking while talking (WT) to classify HOA/MS sequences and subjects. We use majority voting for subject classification. 

* **SubjectGeneralize_MLTraditional.ipynb**: Traditional ML algorithms on subject generalization framework using cross validation to classify HOA/MS sequences and subjects. We use majority voting for subject classification. 

* **Runner.py**: A runner file for all deep learning algorithms on task and subject generalization frameworks to classify HOA and PwMS strides and subjects.

* **config_files/** contains configuration templates to optimize hyperparamaters for the deep learning models for main classification results as well as ablation results

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

### Citation:
If you use this code, please consider citing our work:

(1)
```
```
