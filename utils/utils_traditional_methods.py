from importlib import reload
import utils.package_imports
reload(utils.package_imports)
from utils.package_imports import *
import xgboost

#Summary statistics generation utility functions 
def read_data(data_path, drop_cols = True):
    '''
    Function to read and return relevant data from path
    Arguments: 
        data_path: Path to data's csv file
        drop_cols: True if unrelevant columns need to be dropped, False otherwise
    Returns: 
        Dataframe with relevant data columns 
    '''
    data = pd.read_csv(data_path, index_col=0)
    data.reset_index(inplace= True)
    data = data.drop('index', axis = 1)
    if drop_cols:
        data = data.drop(['tspeed_HSR', 'tspeed_MidSSR', 'tspeed_TOR', 'tspeed_HSL', 'tspeed_TOL', 'tspeed_MidSSL', 
               'Butterfly_y_abs', 'ButterflySQ_y', 'SS_L'], axis = 1)
    #Creates a unique ID for each subject, since PID numbers aren't unique
    data["id"] = data.PID.astype(str) + data.TrialID.astype(str)
    return data


def genarate_sequences(data, strides_per_sequence, skipped_steps):
    '''
    Function to generate sequence with features from multiple strides 
    Arguments:
        stride_per_sequence: Number of multiple strides to collect in a sequence 
        skipped_steps: Number of strides to skip when making the next sequence of stride_per_sequence group of strides 
        This is similar to a moving window of length stride_per_sequence that moves ahead skipped_steps each time 
    Returns:
        seqs: list of dataframes where each dataframe is a group of 10 strides with it's corresponding 
              21 gait features, PID, trialID, unique ID and label. Hence, each dataframe in seqs is shaped (10, 25). 
    
    '''
    row = 0
    nullseq = 0;
    id = data.loc[0, 'id']
    pid = data.loc[0, 'PID']
    seq_count = 0
    seqs = [] #List to collect grouped dataframes of multiple strides 

    while row < (data.shape[0] - strides_per_sequence):
        if data.loc[row, 'id'] == id and data.loc[row+strides_per_sequence, 'id'] == id:
            seq = data.iloc[row:row+strides_per_sequence].drop(['PID', 'Label', 'id'], axis = 1)
            if not seq.isnull().values.any():
                #print(type(data.iloc[row:row+strides_per_sequence]))
                seqs.append(data.iloc[row:row+strides_per_sequence])
                row +=skipped_steps
            else:
                nullseq+=1
                row+=1
            
        else:
            id = data.loc[row, 'id']
            pid = data.loc[row,'PID']
            row+=1
    
    print("dropped sequences: ", nullseq)
    #print(seqs)
    return seqs


def generate_summary_dataframe(sequence_data):
    '''
    Function to construct the dataframe with summary statistics for each sequence of multiple strides. This dataframe 
    consists of features that will be used for traditional algorithms.
    Arguments:
        sequence_data: The list of grouped dataframes of multiple strides 
    Returns:
        summary_dataframe: A dataframe with summary statistics for each sequence of multiple strides
    
    '''
    #Extracting the 21 gait feature names 
    features = sequence_data[0].columns.drop(['PID', 'TrialID', 'id', 'Label'])
    #Summary statistics are mean and standard deviation of the 21 gait features 
    summary_features = list(features+'_mean') + list(features+'_std')
    #Attaching the PID, trialID, count of sequence for a particula PID and trialID and label to the summary features 
    summary_columns = summary_features + ['PID', 'TrialID', 'SeqNum', 'Label']
    summary_dataframe = pd.DataFrame(columns = summary_columns)
    
    idCount = 0
    id_ = sequence_data[0].loc[1, 'id']
    #Iterating through each dataframe in the list of dataframes 
    for idx, data in enumerate(sequence_data):
        data.reset_index(inplace= True)
        #print(data.head())
        #print(data.loc[0, 'id'])
        if (data.loc[0, 'id'] == id_):
            idCount += 1
        else:
            idCount = 0
            id = data.loc[0, 'id']
        summary_dataframe.loc[idx, 'PID'] = data.loc[0, 'PID']
        summary_dataframe.loc[idx, 'TrialID'] = data.loc[0, 'TrialID']
        summary_dataframe.loc[idx, 'SeqNum'] = idCount
        summary_dataframe.loc[idx, 'Label'] = data.loc[0, 'Label']
        data_drop = data.drop(['PID', 'Label', 'TrialID', 'id', 'index'], axis = 1)
        #Appending the mean and standard deviation for the multiple strides groups 
        summary_dataframe.loc[idx, data_drop.columns + '_mean'] = data_drop.mean().values
        summary_dataframe.loc[idx, data_drop.columns + '_std'] = data_drop.std().values
    return summary_dataframe


#Task generalization utility functions 
def normalize(data, n_type):
    '''
    Function to compute the coefficients from the training data to normalize the training and test data sets 
    Arguments: 
        data: dataframe to normalize
        n_type: type of normalization (z-score or min-max)
    Returns:
        Coefficients such that normalized_data = (data - mean)/sd
        mean: mean/min of the training data for z-score/min-max normalization respectively
        sd: standard deviation/max-min of the training data for z-score/min-max normalization respectively
    '''
    #col_names = list(dataframe.columns)
    if (n_type == 'z'): #z-score normalization
        mean = np.mean(data)
        sd = np.std(data)
    else: #min-max normalization
        mean = np.min(data)
        sd = np.max(data)-np.min(data)
    return mean, sd


#Summary statistics for the sequence of multiple strides in raw data 
def extract_features_labels_task_generalize(summary_csv_path):
    '''
    Function to extract training and testing features/X and labels/Y for task generalization using traditional models.
    We read, shuffle and normalize the data.
    Arguments:
        summary_csv_path: The csv file path for the summary statistics of the data type (raw/size-N/regress-N) of interest
    Returns:
        trainX: Features/X for the training set 
        trainY: Labels/Y for the training set 
        testX: Features/X for the training set
        testY: Labels/Y for the training set
    '''
    data = pd.read_csv(summary_csv_path)
    data = data.rename(columns={'Label':"label"})
    data_grouped = data.groupby("TrialID")
    #For task generalization, training set is trial W
    train = data_grouped.get_group(1)
    #Shuffling the training data 
    train = shuffle(train, random_state=0)
    #Testing set is trial WT
    test = data_grouped.get_group(2)
    #Shuffling the testing data 
    test = shuffle(test, random_state=0)
    #Extracting the X/features and Y/labels for the training and testing sets 
    trainXraw = train.drop(['PID', 'label', 'TrialID', 'SeqNum'], axis = 1)
    trainY = train.filter(['PID','label'], axis=1)
    testXraw = test.drop(['PID', 'label', 'TrialID', 'SeqNum'], axis = 1)
    testY = test.filter(['PID','label'], axis=1)

    #z score normalize columns with training data means and std
    mean, sd = normalize(trainXraw, 'z')
    #Normalized trian features 
    trainX = (trainXraw-mean)/sd
    #Normalized test features 
    testX = (testXraw-mean)/sd
    print ('TrainX shape: ', trainX.shape)
    display(trainX.head())
    print ('TrainY shape: ', trainY.shape)
    print ('TestX shape: ', testX.shape)
    display(testX.head())
    print ('TestY shape: ', testY.shape)
    return trainX, trainY, testX, testY


def evaluate_task_generalize(model, test_features, trueY, model_name = 'random_forest', data_type = 'raw_data', results_dir = '5strides/', datastream_name = 'All', save_results = True):
    '''
    Function to test and evaluate the tuned ML models for task generalization (train on W-> test on WT) over sequence and subejct
    based evaluation metrics (accuracy, Precision, Recall, F1 score and AUC).
    Further, it plots and saves the sequence and subject wise confusion matrices for the model and data type of interest.
    Arguments:
        model: grid search model with optimal hyperparameters tuned 
        test_features: X/features for the testing data 
        trueY: Y/labels and PID for the testing data
        model_name: name of the model to save the confusion matrix for 
        data_type: raw/sizeN/regressN_data to save the confusion matrix for 
        save_results: Whether to save the results or not
    Returns:
        proportion_strides_correct['prob_class1']: Prediction probabilities for HOA/MS for the ROC curve
        [acc, p, r, f1, auc, person_acc, person_p, person_r, person_f1, person_auc]: sequence and subject wise 
                                                   evaluation metrics (Accuracy, Precision, Recall, F1 and AUC)
    '''
    test_labels = trueY['label'] #Dropping the PID
    predictions = model.predict(test_features)
    try:
        prediction_prob = model.predict_proba(test_features)[:, 1] #Score of the class with greater label
    except:
        prediction_prob = model.best_estimator_._predict_proba_lr(test_features)[:, 1] #For linear SVM 
    #Stride wise metrics 
    acc = accuracy_score(test_labels, predictions)
    p = precision_score(test_labels, predictions)
    r = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    auc = roc_auc_score(test_labels, prediction_prob)
    print('Stride-based model performance: ', acc, p, r, f1, auc)
    m = (acc, p, r, f1, auc)
    #return m
    
    #For computing person wise metrics 
    temp = copy.deepcopy(trueY) #True label for the stride 
    temp['pred'] = predictions #Predicted label for the stride 
    #Correctly slassified strides i.e. 1 if stride is correctly classified and 0 if otherwise
    temp['correct'] = (temp['label']==temp['pred'])
    #Proportion of correctly classified strides
    proportion_strides_correct = temp.groupby('PID').aggregate({'correct': 'mean'})  
    proportion_strides_correct['True Label'] = trueY.groupby('PID').first() 
    #Label for the person - 0=healthy, 1=MS patient
    proportion_strides_correct['Predicted Label'] = proportion_strides_correct['True Label']*\
    (proportion_strides_correct['correct']>0.5)+(1-proportion_strides_correct['True Label'])*\
    (proportion_strides_correct['correct']<0.5) 
    #Probability of class 1 - MS patient for AUC calculation
    proportion_strides_correct['prob_class1'] = (1-proportion_strides_correct['True Label'])*\
    (1-proportion_strides_correct['correct'])+ proportion_strides_correct['True Label']*proportion_strides_correct['correct'] 
    try:
        print (model.best_estimator_)
    except:
        pass
    #Person wise metrics 
    person_acc = accuracy_score(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'])
    person_p = precision_score(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'])
    person_r = recall_score(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'])
    person_f1 = f1_score(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'])
    person_auc = roc_auc_score(proportion_strides_correct['True Label'], proportion_strides_correct['prob_class1'])
    print('Person-based model performance: ', person_acc, person_p, person_r, person_f1, person_auc)
    
    #Plotting and saving the sequence and subject wise confusion matrices 
    #Sequence wise confusion matrix
    plt.figure()
    confusion_matrix = pd.crosstab(temp['label'], temp['pred'], \
                                   rownames=['Actual'], colnames=['Predicted'], margins = True)
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", fmt = 'd')
    if save_results:
        plt.savefig(results_path_task_generalize_trad + results_dir +'CFmatrix_task_generalize_' + str(data_type) + '_'+ str(model_name) + '_' + datastream_name + '_seq_wise.png', dpi = 350)
    plt.show()
    
    #Subject wise confusion matrix
    plt.figure()
    confusion_matrix = pd.crosstab(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'], \
                                   rownames=['Actual'], colnames=['Predicted'], margins = True)
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", fmt = 'd')
    if save_results:
        plt.savefig(results_path_task_generalize_trad + results_dir + 'CFmatrix_task_generalize_' + str(data_type) + '_'+ str(model_name) + '_' + datastream_name + '_subject_wise.png', dpi = 350)
    plt.show()
    
    return proportion_strides_correct['prob_class1'], [acc, p, r, f1, auc, person_acc, person_p, person_r, person_f1, person_auc]
    
    
def models_task_generalize(trainX, trainY, testX, testY, model_name = 'random_forest', data_type = 'raw_data', results_dir = '5strides/', datastream_name = 'All', save_results = True):
    '''
    Function to define and tune ML models for task generalization (train on W-> test on WT)
    Arguments: 
        trainX, testX: training set
        testX, testY: testing set
        model_name: model
        data_type: raw/sizeN/regressN_data to signify the data type 
        save_results: Whether to save the results or not
    Returns: 
        person_wise_prob_for_roc: Prediction probabilities for HOA/MS 
        stride_person_metrics: sequence and subject wise evaluation metrics (Accuracy, Precision, Recall, F1 and AUC)
    Make sure the sequences of same subject do not appear in both training and validation sets made out of trial W
    '''
    trainY1 = trainY['label'] #Dropping the PID
    #Make sure subjects are not mixed in training and validation sets, i.e. strides of same subject are either 
    #in training set or in validation set 
    groups_ = trainY['PID'] 
    #We use group K-fold to sample our data
    gkf = GroupKFold(n_splits=5) 
    if(model_name == 'random_forest'): #Random Forest
        grid = {
       'n_estimators': [40,45,50],\
       'max_depth' : [15,20,25,None],\
       'class_weight': [None, 'balanced'],\
       'max_features': ['auto','sqrt','log2', None],\
       'min_samples_leaf':[1,2,0.1,0.05]
        }
        rf_grid = RandomForestClassifier(random_state=0)
        #Make sure the strides of same subject do not appear in both training and validation sets made out of trial W
        grid_search = GridSearchCV(estimator = rf_grid, param_grid = grid, scoring='accuracy', n_jobs = 1, \
                                   cv=gkf.split(trainX, trainY1, groups=groups_))
    if(model_name == 'adaboost'): #Adaboost
        ada_grid = AdaBoostClassifier(random_state=0)
        grid = {
        'n_estimators':[50, 75, 100, 125, 150],\
        'learning_rate':[0.01,.1, 1, 1.5, 2]\
        }
        grid_search = GridSearchCV(ada_grid, param_grid = grid, scoring='accuracy', n_jobs = 1, \
                                   cv=gkf.split(trainX, trainY1, groups=groups_))
    if(model_name == 'kernel_svm'): #RBF SVM
        svc_grid = SVC(kernel = 'rbf', probability=True, random_state=0)
        grid = {
        'gamma':[0.0001, 0.001, 0.1, 1, 10, ]\
        }
        grid_search = GridSearchCV(svc_grid, param_grid=grid, scoring='accuracy', n_jobs = 1, \
                                  cv=gkf.split(trainX, trainY1, groups=groups_))
    if(model_name == 'gbm'): #GBM
        gbm_grid = GradientBoostingClassifier(random_state=0)
        grid = {
        'learning_rate':[0.15,0.1,0.05], \
        'n_estimators':[50, 100, 150],\
        'max_depth':[2,4,7],\
        'min_samples_split':[2,4], \
        'min_samples_leaf':[1,3],\
        'max_features':[4, 5, 6]\
        }
        grid_search = GridSearchCV(gbm_grid, param_grid=grid, scoring='accuracy', n_jobs = 1, \
                                  cv=gkf.split(trainX, trainY1, groups=groups_))
    if(model_name=='xgboost'): #Xgboost
        xgb_grid = xgboost.XGBClassifier(random_state=0)
        grid = {
            'min_child_weight': [1, 5],\
            'gamma': [0.1, 0.5, 1, 1.5, 2],\
            'subsample': [0.6, 0.8, 1.0],\
            'colsample_bytree': [0.6, 0.8, 1.0],\
            'max_depth': [5, 7, 8]
        }
        grid_search = GridSearchCV(xgb_grid, param_grid=grid, scoring='accuracy', n_jobs = 1, \
                                  cv=gkf.split(trainX, trainY1, groups=groups_))
    if(model_name == 'knn'): #KNN
        knn_grid = KNeighborsClassifier()
        grid = {
            'n_neighbors': [1, 3, 4, 5, 10],\
            'p': [1, 2, 3, 4, 5]\
        }
        grid_search = GridSearchCV(knn_grid, param_grid=grid, scoring='accuracy', n_jobs = 1, \
                                  cv=gkf.split(trainX, trainY1, groups=groups_))
    if(model_name == 'decision_tree'): #Decision Tree
        dec_grid = DecisionTreeClassifier(random_state=0)
        grid = {
            'min_samples_split': range(2, 50),\
        }
        grid_search = GridSearchCV(dec_grid, param_grid=grid, scoring='accuracy', n_jobs = 1, \
                                  cv=gkf.split(trainX, trainY1, groups=groups_))
    if(model_name == 'linear_svm'): #Linear SVM
        lsvm_grid = LinearSVC(random_state=0)
        grid = {
            'loss': ['hinge','squared_hinge'],\
        }
        grid_search = GridSearchCV(lsvm_grid, param_grid=grid, scoring='accuracy', n_jobs = 1, \
                                  cv=gkf.split(trainX, trainY1, groups=groups_))
    if(model_name == 'logistic_regression'): #Logistic regression
        logistic_grid = LogisticRegression(random_state=0)
        grid = {
            'random_state': [0]
        }
        grid_search = GridSearchCV(logistic_grid, param_grid=grid, scoring='accuracy', n_jobs = 1, \
                                  cv=gkf.split(trainX, trainY1, groups=groups_))  
    if(model_name == 'mlp'):
        mlp_grid = MLPClassifier(activation='relu', solver='adam', learning_rate = 'adaptive', learning_rate_init=0.001,\
                                                        shuffle=False, max_iter = 500, random_state = 0)
        grid = {
            'hidden_layer_sizes': [(128, 8, 8, 128, 32), (50, 50, 50, 50, 50, 50, 150, 100, 10), 
                                  (50, 50, 50, 50, 50, 60, 30, 20, 50), (50, 50, 50, 50, 50, 150, 10, 60, 150),
                                  (50, 50, 50, 50, 50, 5, 50, 10, 5), (50, 50, 50, 50, 50, 5, 50, 150, 150),
                                  (50, 50, 50, 50, 50, 5, 30, 50, 20), (50, 50, 50, 50, 10, 150, 20, 20, 30),
                                  (50, 50, 50, 50, 30, 150, 100, 20, 100), (50, 50, 50, 50, 30, 5, 100, 20, 100),
                                  (50, 50, 50, 50, 60, 50, 50, 60, 60), (50, 50, 50, 50, 20, 50, 60, 20, 20),
                                  (50, 50, 50, 10, 50, 10, 150, 60, 150), (50, 50, 50, 10, 50, 150, 30, 150, 5),
                                  (50, 50, 50, 10, 50, 20, 150, 5, 10), (50, 50, 50, 10, 150, 50, 20, 20, 100), 
                                  (50, 50, 50, 30, 100, 5, 30, 150, 30), (50, 50, 50, 50, 100, 150, 100, 200), 
                                  (50, 50, 50, 5, 5, 100, 100, 150), (50, 50, 5, 50, 200, 100, 150, 5), 
                                  (50, 50, 5, 5, 200, 100, 50, 30), (50, 50, 5, 10, 5, 200, 200, 10), 
                                  (50, 50, 5, 30, 5, 5, 50, 10), (50, 50, 5, 200, 50, 5, 5, 50), 
                                  (50, 50,50, 5, 5, 100, 100, 150), (5, 5, 5, 5, 5, 100, 50, 5, 50, 50), 
                                  (5, 5, 5, 5, 5, 100, 20, 100, 30, 30), (5, 5, 5, 5, 5, 20, 20, 5, 30, 100), 
                                  (5, 5, 5, 5, 5, 20, 20, 100, 10, 10), (5, 5, 5, 5, 10, 10, 30, 50, 10, 10), 
                                  (5, 5, 5, 5, 10, 100, 30, 30, 30, 10), (5, 5, 5, 5, 10, 100, 50, 10, 50, 10), 
                                  (5, 5, 5, 5, 10, 100, 20, 100, 30, 5), (5, 5, 5, 5, 30, 5, 20, 30, 100, 50), 
                                  (5, 5, 5, 5, 30, 100, 20, 50, 20, 30), (5, 5, 5, 5, 50, 30, 5, 50, 10, 100), 
                                  (21, 21, 7, 84, 21, 84, 84), (21, 21, 5, 42, 42, 7, 42), (21, 84, 7, 7, 7, 84, 5), 
                                  (21, 7, 84, 5, 5, 21, 120), (42, 5, 21, 21, 21, 5, 120), (42, 5, 42, 84, 7, 120, 84), 
                                  (50, 100, 10, 5, 100, 25), (10, 10, 25, 50, 25, 5), (50, 50, 50, 50, 50, 20, 30, 100, 60)]
        }
        grid_search = GridSearchCV(mlp_grid, param_grid=grid, scoring='accuracy', n_jobs = 1, \
                                  cv=gkf.split(trainX, trainY1, groups=groups_))
    #Making sure that strides of same subjects do not mix in training and validation sets 
    grid_search.fit(trainX, trainY1, groups=groups_) #Fitting on the training set to find the optimal hyperparameters 
#     print('best score: ', grid_search.best_score_)
#     print('best_params: ', grid_search.best_params_, grid_search.best_index_)
#     print('Mean cv accuracy on test set:', grid_search.cv_results_['mean_test_score'][grid_search.best_index_])
#     print('Standard deviation on test set:' , grid_search.cv_results_['std_test_score'][grid_search.best_index_])
#     print('Mean cv accuracy on train set:', grid_search.cv_results_['mean_train_score'][grid_search.best_index_])
#     print('Standard deviation on train set:', grid_search.cv_results_['std_train_score'][grid_search.best_index_])
#     print('Test set performance:\n')
    person_wise_prob_for_roc, stride_person_metrics = evaluate_task_generalize(grid_search, testX, testY, model_name, data_type, results_dir, datastream_name, save_results)
    return person_wise_prob_for_roc, stride_person_metrics


def plotROC_task_generalize(ml_models, test_Y, predicted_probs_person, metrics_personAUC, data_type = 'raw_data', results_dir = '5strides/', datastream_name = 'All', save_results = True):
    '''
    Function to plot and save the ROC curve for models given in ml_models list to the results directory
    Arguments: 
        ml_models: name of models to plot the ROC for 
        test_Y: true test set labels with PID
        predicted_probs_person: predicted test set probabilities
        metrics_personAUC: Subject-wise AUC for the corresponding data type (raw/size-N/regress-N) to label the ROC plot
        data_type: raw/sizeN/regressN_data to plot and save the ROC for
        save_results: Whether to save the results or not 
    '''
    ml_model_names = {'random_forest': 'RF', 'adaboost': 'Adaboost', 'kernel_svm': 'RBF SVM', 'gbm': 'GBM', \
                      'xgboost': 'Xgboost', 'knn': 'KNN', 'decision_tree': 'DT',  'linear_svm': 'LSVM', 
                 'logistic_regression': 'LR', 'mlp': 'MLP'}
    data_names = {'raw_data': 'Raw data', 'sizeN_data': 'Size-N data', 'regressN_data': 'Regress-N data'}
    person_true_labels = test_Y.groupby('PID').first()
    neutral = [0 for _ in range(len(person_true_labels))] # ROC for majority class prediction all the time 

    fig, axes = plt.subplots(1, 1, sharex=True, sharey = True, figsize=(5.2, 3.5))
    sns.despine(offset=0)
    neutral_fpr, neutral_tpr, _ = roc_curve(person_true_labels, neutral) #roc curves
    linestyles = ['-', '-', '-', '-.', '--', '-', '--', '-', '--']
    colors = ['b', 'magenta', 'cyan', 'g',  'red', 'violet', 'lime', 'grey', 'pink']

    axes.plot(neutral_fpr, neutral_tpr, linestyle='--', label='Majority (AUC = 0.5)', linewidth = 3, color = 'k')
    for idx, ml_model in enumerate(ml_models):
        model_probs = predicted_probs_person[ml_model] # person-based prediction probabilities
        fpr, tpr, _ = roc_curve(person_true_labels, model_probs)
        axes.plot(fpr, tpr, label=ml_model_names[ml_model]+' (AUC = '+ str(round(metrics_personAUC[ml_model], 3))
                     +')', linewidth = 3, alpha = 0.8, linestyle = linestyles[idx], color = colors[idx])

    axes.set_ylabel('True Positive Rate')
    axes.set_title('Task generalization: ' + data_names[data_type])
    plt.legend()
    # axes[1].legend(loc='upper center', bbox_to_anchor=(1.27, 1), ncol=1)

    axes.set_xlabel('False Positive Rate')
    plt.tight_layout()
    if (len(ml_models)==1):
        savefig_name = data_type + '_' + str(ml_model)
    else:
        savefig_name = data_type
    if save_results:
        plt.savefig(results_path_task_generalize_trad + results_dir + 'ROC_task_generalize_' + savefig_name + '_' + datastream_name + '.png', dpi = 250)
    plt.show()
    
    
#Subject generalization utility functions 
def extract_data_subject_generalize(summary_csv_path):
    '''
    Function to extract features/X and labels/Y for subject generalization using traditional models.
    We read and shuffle the data.
    Arguments:
        summary_csv_path: The csv file path for the summary statistics of the data type (raw/size-N/regress-N) of interest
    Returns:
        X: Features/X for the data type of interest
        Y:Labels/Y for the data type of interest
    '''
    data = pd.read_csv(summary_csv_path)
    data = data.rename(columns={'Label':"label"})
    #Shuffling the data
    data = shuffle(data, random_state=0)
    #Extracting the X/features and Y/labels for the training and testing sets 
    X = data.drop(['PID', 'label', 'TrialID', 'SeqNum'], axis = 1)
    Y = data.filter(['PID','label'], axis=1)
    print ('X shape: ', X.shape)
    display(X.head())
    print ('Y shape: ', Y.shape)
    return X, Y


def acc(y_true, y_pred):
    '''
    Returns the accuracy 
    Saves the true and predicted labels for training and test sets
    '''
    global yoriginal, ypredicted
    yoriginal.append(y_true)
    ypredicted.append(y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def evaluate_subject_generalize(model, test_features, yoriginal_, ypredicted_, model_name = 'random_forest', data_type = 'raw_data', results_dir = '5strides/', datastream_name = 'All', save_results = True):
    '''
    Function to test and evaluate the tuned ML models for subject generalization over sequence and subejct
    based evaluation metrics (accuracy, Precision, Recall, F1 score and AUC).
    Further, it plots and saves the sequence and subject wise confusion matrices for the model and data type of interest.
    Arguments: 
        model: trained grid search model with optimal hyperparameters tuned 
        test_features: X/features for the testing data 
        yoriginal_: true labels for test set
        ypredicted_: predicted labels for the test set
        model_name: model name to save the confusion matrix for 
        data_type: raw/sizeN/regressN_data to save the confusion matrix for
        results_dir: Depends on how many strides per sequence were selected for computing summary statistics
        save_results: Whether to save the results or not 
    Returns: 
        tpr_list: True positive rate for the ROC curve 
        [stride_metrics_mean, stride_metrics_std, person_means, person_stds]: Means and standard deviations for the 
                                                                        sequence and subject based evaluation metrics 
    '''      
   
    #For creating the stride wise confusion matrix, we append the true and predicted labels for strides in each fold to this 
    #test_strides_true_predicted_labels dataframe 
    test_strides_true_predicted_labels = pd.DataFrame()
    #For creating the subject wise confusion matrix, we append the true and predicted labels for subjects in each fold to this
    #test_subjects_true_predicted_labels dataframe
    test_subjects_true_predicted_labels = pd.DataFrame()
        
    best_index = model.cv_results_['mean_test_accuracy'].argmax()
    print('best_params: ', model.cv_results_['params'][best_index])

    #Stride-wise metrics 
    stride_metrics_mean, stride_metrics_std = [], [] #Mean and SD of stride based metrics - Acc, P, R, F1, AUC (in order)
    scores={'accuracy': make_scorer(acc), 'precision':'precision', 'recall':'recall', 'f1': 'f1', 'auc': 'roc_auc'}
    for score in scores:
        stride_metrics_mean.append(model.cv_results_['mean_test_'+score][best_index])
        stride_metrics_std.append(model.cv_results_['std_test_'+score][best_index])
    print('Stride-based model performance (mean): ', stride_metrics_mean)
    print('Stride-based model performance (standard deviation): ', stride_metrics_std)
    n_folds = 5
    person_acc, person_p, person_r, person_f1, person_auc = [], [], [], [], []
    #For ROC curves 
    tpr_list = []
    base_fpr = np.linspace(0, 1, 101)

    for i in range(n_folds):
        #For each fold, there are 2 splits: test and train (in order) and we need to retrieve the index 
        #of only test set for required 5 folds (best index)
        temp = test_features.loc[yoriginal_[(best_index*n_folds) + (i)].index] #True labels for the test strides in each fold
#         print ('temp', temp)
#         print (yoriginal_[(best_index*n_folds) + (i)])
        temp['pred'] = ypredicted_[(best_index*n_folds) + (i)] #Predicted labels for the strides in the test set in each fold

        #Correctly classified strides i.e. 1 if stride is correctly classified and 0 if otherwise
        temp['correct'] = (temp['label']==temp['pred'])
        
        #Appending the test strides' true and predicted label for each fold to compute stride-wise confusion matrix 
        test_strides_true_predicted_labels = test_strides_true_predicted_labels.append(temp)
        
        #Proportion of correctly classified strides
        proportion_strides_correct = temp.groupby('PID').aggregate({'correct': 'mean'})  

        proportion_strides_correct['True Label'] = temp[['PID', 'label']].groupby('PID').first() 

        #Label for the person - 0=healthy, 1=MS patient
        proportion_strides_correct['Predicted Label'] = proportion_strides_correct['True Label']*\
        (proportion_strides_correct['correct']>0.5)+(1-proportion_strides_correct['True Label'])*\
        (proportion_strides_correct['correct']<0.5) 

        #Probability of class 1 - MS patient for AUC calculation
        proportion_strides_correct['prob_class1'] = (1-proportion_strides_correct['True Label'])*\
        (1-proportion_strides_correct['correct'])+ proportion_strides_correct['True Label']*proportion_strides_correct['correct'] 
        
        #Appending the test subjects' true and predicted label for each fold to compute subject-wise confusion matrix 
        test_subjects_true_predicted_labels = test_subjects_true_predicted_labels.append(proportion_strides_correct)  
        
        fpr, tpr, _ = roc_curve(proportion_strides_correct['True Label'], proportion_strides_correct['prob_class1'])
        tpr = interp(base_fpr, fpr, tpr)
        tpr[0] = 0.0
        tpr_list.append(tpr)
        #Person wise metrics for each fold 
        person_acc.append(accuracy_score(proportion_strides_correct['Predicted Label'], proportion_strides_correct['True Label']))
        person_p.append(precision_score(proportion_strides_correct['Predicted Label'], proportion_strides_correct['True Label']))
        person_r.append(recall_score(proportion_strides_correct['Predicted Label'], proportion_strides_correct['True Label']))
        person_f1.append(f1_score(proportion_strides_correct['Predicted Label'], proportion_strides_correct['True Label']))
        person_auc.append(roc_auc_score(proportion_strides_correct['True Label'], proportion_strides_correct['prob_class1']))

    #Mean and standard deviation for person-based metrics 
    person_means = [np.mean(person_acc), np.mean(person_p), np.mean(person_r), np.mean(person_f1), np.mean(person_auc)]
    person_stds = [np.std(person_acc), np.std(person_p), np.std(person_r), np.std(person_f1), np.std(person_auc)]
    print('Person-based model performance (mean): ', person_means)
    print('Person-based model performance (standard deviation): ', person_stds)

    #Plotting and saving the sequence and subject wise confusion matrices 
    #Sequence wise confusion matrix
    plt.figure()
    confusion_matrix = pd.crosstab(test_strides_true_predicted_labels['label'], test_strides_true_predicted_labels['pred'], \
                                   rownames=['Actual'], colnames=['Predicted'], margins = True)
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", fmt = 'd')
    if save_results:
        plt.savefig(results_path_subject_generalize_trad + results_dir + 'CFmatrix_subject_generalize_' + str(data_type) + '_'+ str(model_name) + '_' + datastream_name +'_seq_wise.png', dpi = 350)
    plt.show()

    #Subject wise confusion matrix
    plt.figure()
    confusion_matrix = pd.crosstab(test_subjects_true_predicted_labels['True Label'], test_subjects_true_predicted_labels['Predicted Label'], \
                                   rownames=['Actual'], colnames=['Predicted'], margins = True)
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", fmt = 'd')
    if save_results:
        plt.savefig(results_path_subject_generalize_trad +  results_dir + 'CFmatrix_subject_generalize_' + str(data_type) + '_'+ str(model_name) + '_' + datastream_name + '_subject_wise.png', dpi = 350)
    plt.show()
    return tpr_list, [stride_metrics_mean, stride_metrics_std, person_means, person_stds]



#We do not use LDA/QDA since our features are not normally distributed 
def models_subject_generalize(X, Y, model_name = 'random_forest', data_type = 'raw_data', results_dir = '5strides/', datastream_name = 'All', save_results = True):
    '''
    Function to define and tune ML models for subject generalization
    Arguments: 
        X: Feature set for the data type of interest (raw/size-N/regress-N) 
        Y: Labels for the data type of interest along with PID groups so that strides of each person are either in 
        training or in testing set
        model_name: model
        data_type: raw/sizeN/regressN_data to signify the data type 
        results_dir: Depends on how many strides per sequence were selected for computing summary statistics
        save_results: Whether to save the results or not
    Returns: 
        tpr_list: True positive rate for the ROC curve 
        stride_person_metrics: Means and standard deviations for the sequence and subject based evaluation metrics 
    '''      
        
    Y_ = Y['label'] #Dropping the PID
    groups_ = Y['PID']
    gkf = GroupKFold(n_splits=5) 
    scores={'accuracy': make_scorer(acc), 'precision':'precision', 'recall':'recall', 'f1': 'f1', 'auc': 'roc_auc'}
    
    if(model_name == 'random_forest'): #Random Forest
        grid = {
       'randomforestclassifier__n_estimators': [40,45,50],\
       'randomforestclassifier__max_depth' : [15,20,25,None],\
       'randomforestclassifier__class_weight': [None, 'balanced'],\
       'randomforestclassifier__max_features': ['auto','sqrt','log2', None],\
       'randomforestclassifier__min_samples_leaf':[1,2,0.1,0.05]
        }
        #For z-score scaling on training and use calculated coefficients on test set
        rf_grid = make_pipeline(StandardScaler(), RandomForestClassifier(random_state=0))
        grid_search = GridSearchCV(rf_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)
    
    if(model_name == 'adaboost'): #Adaboost
        ada_grid = make_pipeline(StandardScaler(), AdaBoostClassifier(random_state=0))
        grid = {
        'adaboostclassifier__n_estimators':[50, 75, 100, 125, 150],\
        'adaboostclassifier__learning_rate':[0.01,.1, 1, 1.5, 2]\
        }
        grid_search = GridSearchCV(ada_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)
        
    if(model_name == 'kernel_svm'): #RBF SVM
        svc_grid = make_pipeline(StandardScaler(), SVC(kernel = 'rbf', probability=True, random_state=0))
        grid = {
        'svc__gamma':[0.0001, 0.001, 0.1, 1, 10, ]\
        }
        grid_search = GridSearchCV(svc_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)

    if(model_name == 'gbm'): #GBM
        gbm_grid = make_pipeline(StandardScaler(), GradientBoostingClassifier(random_state=0))
        grid = {
        'gradientboostingclassifier__learning_rate':[0.15,0.1,0.05], \
        'gradientboostingclassifier__n_estimators':[50, 100, 150],\
        'gradientboostingclassifier__max_depth':[2,4,7],\
        'gradientboostingclassifier__min_samples_split':[2,4], \
        'gradientboostingclassifier__min_samples_leaf':[1,3],\
        'gradientboostingclassifier__max_features':['auto','sqrt','log2', None],\
        }
        grid_search = GridSearchCV(gbm_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)
    
    if(model_name=='xgboost'): #Xgboost
        xgb_grid = make_pipeline(StandardScaler(), xgboost.XGBClassifier(random_state=0))
        grid = {
            'xgbclassifier__min_child_weight': [1, 5],\
            'xgbclassifier__gamma': [0.1, 0.5, 1, 1.5, 2],\
            'xgbclassifier__subsample': [0.6, 0.8, 1.0],\
            'xgbclassifier__colsample_bytree': [0.6, 0.8, 1.0],\
            'xgbclassifier__max_depth': [5, 7, 8]
        }
        grid_search = GridSearchCV(xgb_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)
    
    if(model_name == 'knn'): #KNN
        knn_grid = make_pipeline(StandardScaler(), KNeighborsClassifier())
        grid = {
            'kneighborsclassifier__n_neighbors': [1, 3, 4, 5, 10],\
            'kneighborsclassifier__p': [1, 2, 3, 4, 5]\
        }
        grid_search = GridSearchCV(knn_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)
        
    if(model_name == 'decision_tree'): #Decision Tree
        dec_grid = make_pipeline(StandardScaler(), DecisionTreeClassifier(random_state=0))
        #For z-score scaling on training and use calculated coefficients on test set
        grid = {'decisiontreeclassifier__min_samples_split': range(2, 50)}
        grid_search = GridSearchCV(dec_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)

    if(model_name == 'linear_svm'): #Linear SVM
        lsvm_grid = make_pipeline(StandardScaler(), LinearSVC(random_state=0))
        grid = {
            'linearsvc__loss': ['hinge','squared_hinge'],\

        }
        grid_search = GridSearchCV(lsvm_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)
    if(model_name == 'mlp'):
        mlp_grid = make_pipeline(StandardScaler(), MLPClassifier(random_state = 0, activation='relu', solver='adam',\
                                                       learning_rate = 'adaptive', learning_rate_init=0.001, 
                                                        shuffle=False, max_iter = 200))
        grid = {
            'mlpclassifier__hidden_layer_sizes': [(128, 8, 8, 128, 32), (50, 50, 50, 50, 50, 50, 150, 100, 10), 
                                  (50, 50, 50, 50, 50, 60, 30, 20, 50), (50, 50, 50, 50, 50, 150, 10, 60, 150),
                                  (50, 50, 50, 50, 50, 5, 50, 10, 5), (50, 50, 50, 50, 50, 5, 50, 150, 150),
                                  (50, 50, 50, 50, 50, 5, 30, 50, 20), (50, 50, 50, 50, 10, 150, 20, 20, 30),
                                  (50, 50, 50, 50, 30, 150, 100, 20, 100), (50, 50, 50, 50, 30, 5, 100, 20, 100),
                                  (50, 50, 50, 50, 60, 50, 50, 60, 60), (50, 50, 50, 50, 20, 50, 60, 20, 20),
                                  (50, 50, 50, 10, 50, 10, 150, 60, 150), (50, 50, 50, 10, 50, 150, 30, 150, 5),
                                  (50, 50, 50, 10, 50, 20, 150, 5, 10), (50, 50, 50, 10, 150, 50, 20, 20, 100), 
                                  (50, 50, 50, 30, 100, 5, 30, 150, 30), (50, 50, 50, 50, 100, 150, 100, 200), 
                                  (50, 50, 50, 5, 5, 100, 100, 150), (50, 50, 5, 50, 200, 100, 150, 5), 
                                  (50, 50, 5, 5, 200, 100, 50, 30), (50, 50, 5, 10, 5, 200, 200, 10), 
                                  (50, 50, 5, 30, 5, 5, 50, 10), (50, 50, 5, 200, 50, 5, 5, 50), 
                                  (50, 50,50, 5, 5, 100, 100, 150), (5, 5, 5, 5, 5, 100, 50, 5, 50, 50), 
                                  (5, 5, 5, 5, 5, 100, 20, 100, 30, 30), (5, 5, 5, 5, 5, 20, 20, 5, 30, 100), 
                                  (5, 5, 5, 5, 5, 20, 20, 100, 10, 10), (5, 5, 5, 5, 10, 10, 30, 50, 10, 10), 
                                  (5, 5, 5, 5, 10, 100, 30, 30, 30, 10), (5, 5, 5, 5, 10, 100, 50, 10, 50, 10), 
                                  (5, 5, 5, 5, 10, 100, 20, 100, 30, 5), (5, 5, 5, 5, 30, 5, 20, 30, 100, 50), 
                                  (5, 5, 5, 5, 30, 100, 20, 50, 20, 30), (5, 5, 5, 5, 50, 30, 5, 50, 10, 100), 
                                  (21, 21, 7, 84, 21, 84, 84), (21, 21, 5, 42, 42, 7, 42), (21, 84, 7, 7, 7, 84, 5), 
                                  (21, 7, 84, 5, 5, 21, 120), (42, 5, 21, 21, 21, 5, 120), (42, 5, 42, 84, 7, 120, 84), 
                                  (50, 100, 10, 5, 100, 25), (10, 10, 25, 50, 25, 5), (50, 50, 50, 50, 50, 20, 30, 100, 60)]

        }
        grid_search = GridSearchCV(mlp_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)
        
    if(model_name == 'logistic_regression'): #Logistic regression
        lr_grid = make_pipeline(StandardScaler(), LogisticRegression())
        grid = {
            'logisticregression__random_state': [0]}
            
        grid_search = GridSearchCV(lr_grid, param_grid=grid, scoring=scores\
                           , n_jobs = 1, cv=gkf.split(X, Y_, groups=groups_), refit=False)
    
    grid_search.fit(X, Y_, groups=groups_) #Fitting on the training set to find the optimal hyperparameters 
#     print ('y_original', yoriginal)
#     print ('y_predicted', ypredicted)
    tpr_list, stride_person_metrics = evaluate_subject_generalize(grid_search, Y, yoriginal, ypredicted, model_name, data_type, results_dir, datastream_name, save_results)
    return tpr_list, stride_person_metrics


#ROC curves for subject generalization framework
def plot_ROC_subject_generalize(ml_models, tprs_list, metrics_personAUC, data_type = 'raw_data', results_dir = '5strides/', datastream_name = 'All', save_results = True):
    '''
    Function to plot and save the ROC curve for subject generalization model given in ml_models list to the results directory
    Input: 
        ml_models: name of models to plot the ROC for 
        tprs_list: tprs list for all models of interest and the data type of interest
        metrics_personAUC: Subject-wise AUC for the corresponding data type (raw/size-N/regress-N) to label the ROC plot
        data_type: raw/sizeN/regressN_data to plot and save the ROC for
        results_dir: Depends on how many strides per sequence were selected for computing summary statistics
        save_results: Whether to save the results or not 
    '''
    
    base_fpr = np.linspace(0, 1, 101)
    ml_model_names = {'random_forest': 'RF', 'adaboost': 'AdaBoost', 'kernel_svm': 'RBF SVM', 'gbm': 'GBM', \
                      'xgboost': 'Xgboost', 'knn': 'KNN', 'decision_tree': 'DT',  'linear_svm': 'LSVM', 
                 'logistic_regression': 'LR', 'mlp':'MLP'}
    data_names = {'raw_data': 'Raw data', 'sizeN_data': 'Size-N data', 'regressN_data': 'Regress-N data'}

    fig, axes = plt.subplots(1, 1, sharex=True, sharey = True, figsize=(5.2, 3.5))
    sns.despine(offset=0)

    linestyles = ['-', '-', '-', '-.', '--', '-', '--', '-', '--']
    colors = ['b', 'magenta', 'cyan', 'g',  'red', 'violet', 'lime', 'grey', 'pink']

    axes.plot([0, 1], [0, 1], linestyle='--', label='Majority (AUC = 0.5)', linewidth = 3, color = 'k')
    for idx, ml_model in enumerate(ml_models):
        tprs = tprs_list[ml_model] # person-based prediction probabilities
        tprs = np.array(tprs)
        mean_tprs = tprs.mean(axis=0)
        std = tprs.std(axis=0)

        tprs_upper = np.minimum(mean_tprs + std, 1)
        tprs_lower = mean_tprs - std
    #     axes[2].fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
        axes.plot(base_fpr, mean_tprs, label=ml_model_names[ml_model]+' (AUC = '+ str(round(metrics_personAUC.loc['person_mean_AUC']
                         [ml_model], 2)) + r'$\pm$' + str(round(metrics_personAUC.loc['person_std_AUC']
                         [ml_model], 2)) + ')', linewidth = 3, alpha = 0.8, linestyle = linestyles[idx], color = colors[idx])
    axes.set_ylabel('True Positive Rate')
    axes.set_title('Subject generalization: ' + data_names[data_type])
    axes.legend() #loc='upper center', bbox_to_anchor=(1.27, 1), ncol=1)

    axes.set_xlabel('False Positive Rate')
    plt.tight_layout()
    if (len(ml_models)==1):
        savefig_name = data_type + '_' + str(ml_model)
    else:
        savefig_name = data_type
    if save_results:
        plt.savefig(results_path_subject_generalize_trad + results_dir + 'ROC_subject_generalize_' + savefig_name + '_' + datastream_name + '.png', dpi = 250)
    plt.show()
    
def run_ml_models(ml_models, X, Y, data_type, results_dir = '5strides/', datastream_name = 'All', save_results = True):
    '''
    Function to run the subject generalization ML models for the data of interest 
    Arguments: 
        ml_models: model
        X: Feature set for the data type of interest (raw/size-N/regress-N) 
        Y: Labels for the data type of interest along with PID groups so that strides of each person are either in 
        training or in testing set
        data_type: raw/sizeN/regressN_data to signify the data type 
        results_dir: Depends on how many strides per sequence were selected for computing summary statistics
        save_results: Whether to save the csv files or not 
   Returns:
       metrics: Means and standard deviations for the sequence and subject based evaluation metrics
    '''          
            
    metrics = pd.DataFrame(columns = ml_models) #Dataframe to store accuracies for each ML model
    predicted_probs_person = pd.DataFrame(columns = ml_models)
    for ml_model in ml_models:
        print (ml_model)
        global yoriginal, ypredicted
        yoriginal = []
        ypredicted = []
        predict_probs_person, stride_person_metrics = models_subject_generalize(X, Y, ml_model, data_type, results_dir, datastream_name, save_results)
        metrics[ml_model] = sum(stride_person_metrics, [])
        predicted_probs_person[ml_model] = list(predict_probs_person)
        print ('********************************')

    metrics.index = ['sequence_mean_accuracy', 'sequence_mean_precision', 'sequence_mean_recall', 'sequence_mean_F1', \
                         'sequence_mean_AUC', 'sequence_std_accuracy', 'sequence_std_precision', 'sequence_std_recall', 'sequence_std_F1', \
                         'sequence_std_AUC','person_mean_accuracy', 'person_mean_precision', 'person_mean_recall', 'person_mean_F1',\
                         'person_mean_AUC', 'person_std_accuracy', 'person_std_precision', 'person_std_recall', 'person_std_F1',\
                         'person_std_AUC']  
    metrics_personAUC = metrics.loc[['person_mean_AUC', 'person_std_AUC']]
    #Saving the sequence and subject-wise metrics (mean and standard deviation) and 
    #predicted person wise probabilities to csv files 
    if save_results:
        metrics.to_csv(results_path_subject_generalize_trad + results_dir + 'subject_generalize' + '_' + data_type + '_' + str(datastream_name) + '_traditional_result_metrics.csv')
        predicted_probs_person.to_csv(results_path_subject_generalize_trad + results_dir+ 'subject_generalize_' + data_type + '_' + str(datastream_name) + '_traditional_prediction_probs.csv')

    #Plotting the person-wise ROC curves
    plot_ROC_subject_generalize(ml_models, predicted_probs_person, metrics_personAUC, data_type, results_dir, datastream_name, save_results)
    return metrics 