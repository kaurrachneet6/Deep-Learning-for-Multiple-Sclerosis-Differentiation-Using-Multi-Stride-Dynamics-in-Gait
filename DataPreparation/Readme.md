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
