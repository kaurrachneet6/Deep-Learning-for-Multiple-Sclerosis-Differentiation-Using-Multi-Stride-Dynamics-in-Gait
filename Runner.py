from importlib import reload
from typing import NewType

from pandas.core import frame
#from torch.nn.functional import dropout
import utils.utils_lstm
reload(utils.utils_lstm)
from utils.utils_lstm import evaluate_subject_generalize_lstm, evaluate_task_generalize_lstm, genSequences, genTTSequences, normalize, plotROC_task_generalize_LSTM, read_data, create_model, load_model, read_raw_data, save_model, subject_train, task_learning_curve, task_train, custom_StandardScaler, learning_curves, task_gen_perm_imp_main_setup, subject_gen_perm_imp_main_setup, data_stream_num_features
from utils.package_imports import *
import time
import json
import argparse
import os
from utils.LSTM import LSTM
from utils.GRU import GRU
from utils.RNN import RNN
from utils.TCN_model import TCN
from utils.CNN1D import CNN1D
from utils.utils_lstm import torch_StandardScaler
from utils.MSResNet import MSResNet
from utils.MSResNetRaw import MSResNetRaw
from utils.ResNet import ResNet


#Default vars if not provided

dataset = "sizeN" # "sizeN" "raw" "regressN"
strides_per_sequence = 5 # default 5
framework = "task" # "task" "subject"
behavior = "train" # "train" "evaluate"
saved_model_path = "results/task_generalize_lstm/5strides/sizeNsubjectFalse" #used if running evaluate, which uses an already trained model
bidirectional = False
save_model_destination_task = "results/task_generalize_lstm/"
save_model_destination_subject = "results/subject_generalize_lstm/"


input_size = 21 #We have 21 features 
hidden_size1 = 30
num_layers1 = 5
hidden_size2 = 30
num_layers2 = 5
bidirectional1 = True
bidirectional2 = True
pre_out= 50
num_classes = 2
sequence_length = 10
learning_rate = 0.001
batch_size = 100
num_epochs = 500
dropout_ = .2
bi_mult1 = 1
bi_mult2 = 1
single_layer = True
linear_size = 1
use_layernorm = False


#Default params if not provided
'''
params = {
    'lr': [.001, .0001, .00001],
    'optimizer': [torch.optim.Adam],
    #'optimizer__momentum': [0],
    'module__num_layers1': [1],
    'module__hidden_size1': [10, 25, 50, 150],
    'module__num_layers2': [1],
    'module__hidden_size2': [10, 25, 50, 150],
    'module__bidirectional1': [bidirectional],
    'module__bidirectional2': [bidirectional],
    'module__dropout': [0, .3, .6],
    'module__pre_out' : [50],
    'module__input_size': [21],
    'module__num_classes': [2],
    'module__single_layer': [True, False],
    'module__linear_size': [0],
    'module__use_layernorm': [False],
    'module__device': [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
}
'''
params = {}
#1 layer unidirectional and 1 layer bidirectional
#Explore everything else, opitmizers, hidden sizes, etc. keep 1 layer bidirectional.
#Later: 
#Keep second lstm layer parameter
#Parameter to add more another linear fc layer at end
#Use files to set up parameter grids, specified by cli args
'''
subject_params = {
    'net__lr': [.01, .01],
    'net__optimizer': [torch.optim.SGD, torch.optim.Adam],
    'net__module__num_layers1': [10, 40],
    'net__module__hidden_size1': [10, 40],
    'net__module__num_layers2': [10, 40],
    'net__module__hidden_size2': [10, 40],
    'net__module__bidirectional1': [bidirectional],
    'net__module__bidirectional2': [bidirectional],
    'net__module__pre_out' : [50],
    'net__module__dropout': [.4, .7],
    'net__module__input_size': [21],
    'net__module__num_classes': [2],
    'net__module__single_layer': [True, False],
    'net__module__linear_size': [0],
    'net__module__use_layernorm': [False],
    'net__module__device': [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
}
'''

subject_params = {}

#This needs to contain all optimizers that will be used so they can be properly imported
optims = {
    'torch.optim.Adam': torch.optim.Adam,
    'torch.optim.SGD': torch.optim.SGD,
    'torch.optim.Adagrad': torch.optim.Adagrad,
    'torch.optim.AdamW': torch.optim.AdamW,
    'torch.optim.RMSprop': torch.optim.RMSprop,
    "torch.optim.Adamax": torch.optim.Adamax
}

#Set up vars for parsing
vars = {}
parser = argparse.ArgumentParser()
parser.add_argument('config_path', type=str, help='Path to config file')
args = parser.parse_args()

#Load config
with open(args.config_path) as f: 
    config_data = f.read()

config = json.loads(config_data)

#Parse through imported dictionary
isSubject = False
if config["framework"] == "subject":
    isSubject = True

for key, value in config.items():
    if "param" in key:
        if isSubject:
            pkey = key.replace("param", "net__")
            if "net__optimizer" == pkey:
                op = []
                for optim_string in value:
                    op.append(optims[optim_string])
                subject_params[pkey] = op
            elif "net__criterion__weight" == pkey:
                weights = []
                for weight_list in value:
                    weights.append(torch.FloatTensor(weight_list))
                subject_params[pkey] = weights
            else:
                subject_params[pkey] = value
        else:
            pkey = key.replace("param", "")
            if "optimizer" == pkey:
                op = []
                for optim_string in value:
                    op.append(optims[optim_string])
                params[pkey] = op
            else:
                params[pkey] = value
    else:
        vars[key] = value


#Load new vars
dataset = vars["dataset"] # "sizeN" "raw" "regressN"
strides_per_sequence = vars["strides_per_sequence"] # default 5
framework = vars["framework"] # "task" "subject"
behavior = vars["behavior"] # "train" "evaluate"
bidirectional = vars["bidirectional"]
name = vars["name"]
if "model" in vars and (vars["model"] == "RNN" or vars["model"] == "LSTM" or vars["model"] == "GRU"):
    if isSubject:
        subject_params["net__module__bidirectional1"] = [bidirectional]
        subject_params["net__module__bidirectional2"] = [bidirectional]
        subject_params['net__module__device'] = [torch.device("cuda" if torch.cuda.is_available() else "cpu")]
    else:
        params["module__bidirectional1"] = [bidirectional]
        params["module__bidirectional2"] = [bidirectional]    
        params["module__device"] = [torch.device("cuda" if torch.cuda.is_available() else "cpu")]

if "ablation" not in vars:
    vars["ablation"] = "all"

#Create folder for results
time_now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")
path = os.getcwd()
print("path",path)
rpath = ""
if framework == "task":
    rpath = save_model_destination_task
else:
    rpath = save_model_destination_subject
#use this path to save results

#Add ablation + model type to path
#subject_generalize_lstm/5strides/ablation_type/model_type/name+time
#subject_generalize_lstm/5strides/ablation_type/model_type/name+time
if behavior != "feature_importance":
    #add regressN or sizeN to path if not ablation
    if vars["ablation"] == "all":
        lpath = rpath+str(5)+"strides/"+vars["ablation"]+"/"+vars["model"]+"/"+vars["dataset"]+"/"+vars["name"]+time_now+"/"
    else:
        lpath = rpath+str(5)+"strides/"+vars["ablation"]+"/"+vars["model"]+"/"+vars["name"]+time_now+"/"
    print("lpath: ", lpath)
    gpath = path+"/"+lpath
    os.mkdir(gpath)

    #Copy config file to results folder

    with open(args.config_path, 'rb') as src, open(lpath+"/config.json", 'wb') as dst: dst.write(src.read())
vars["config_path"] = args.config_path
#Load dataset
data = None
if dataset == "sizeN":
    data = read_data("data/size_normalized_gait_features.csv")

if dataset == "raw":
    data = read_data("data/gait_features.csv")

if dataset == "regressN":
    data = read_data("data/mr_scaled_features_30controlsTrialW.csv", False)

#if dataset == "raw_features":
    #This is done later
    
    #raw_training_files, raw_training_labels, raw_testing_files, raw_testing_labels = read_raw_data("data/grouped_labels.csv")
    # print("training files: ", training_files)
    # print("training labels: ", training_labels)
    # print("testing files: ", testing_files)
    # print("testing labels: ", testing_labels)

    # print("training files len: ", len(training_files), training_files[0].shape)
    # print("training labels len: ", len(training_labels))
    # print("testing files len: ", len(testing_files), testing_files[0].shape)
    # print("testing labels len: ", len(testing_labels))

    #exit()

#if data != None:
#    print("Data loaded")

if vars["ablation"] != "all":
    if framework == "task":
        if "module__input_size" in params:
            params["module__input_size"] = [data_stream_num_features(vars["ablation"])]
        if "module__in_chans" in params:
            params["module__in_chans"] = [data_stream_num_features(vars["ablation"])]
    if framework == "subject":
        if "net__module__input_size" in subject_params:
            subject_params["net__module__input_size"] = [data_stream_num_features(vars["ablation"])]
        if "net__module__in_chans" in subject_params:
            subject_params["net__module__in_chans"] = [data_stream_num_features(vars["ablation"])]
#net__module__in_chans
#net__module__input_size
#module__input_size
#module__in_chans
print(subject_params)


model = None
    
if framework == "task":
    if "raw_features" not in vars["dataset"]:
        testStridesList, fullTestLabelsList, trainStridesList, fullTrainLabelsList = genTTSequences(data, strides_per_sequence, 2, vars["ablation"])
    else:
        trainStridesList, fullTrainLabelsList, testStridesList, fullTestLabelsList = read_raw_data("data/grouped_labels.csv", True, vars["dataset"])

    print("teststrideslist", np.array(testStridesList).shape)
    print("trainStrideslist", np.array(trainStridesList).shape)
    ss = custom_StandardScaler(strides_per_sequence, np.array(testStridesList).shape[2])
    ss.fit(np.array(testStridesList))
    testStridesList_norm = ss.transform(np.array(testStridesList))
    trainStridesList_norm = ss.transform(np.array(trainStridesList))
    #time_now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%p_%f")
    if behavior == "train":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_model = None
        if vars["model"] == "LSTM":
            input_size = 21
            num_classes = 2
            pre_out = 1
            torch_model = LSTM(input_size, hidden_size1, num_layers1, hidden_size2, num_layers2, num_classes, dropout_, bidirectional, bidirectional, pre_out, single_layer, linear_size, device, use_layernorm).to(device)
        if vars["model"] == "GRU":
            input_size = 21
            num_classes = 2
            pre_out = 1
            use_layernorm = False
            torch_model = GRU(input_size, hidden_size1, num_layers1, hidden_size2, num_layers2, num_classes, dropout_, bidirectional, bidirectional, pre_out, single_layer, linear_size, device, use_layernorm).to(device)
        if vars["model"] == "RNN":
            input_size = 21
            num_classes = 2
            pre_out = 1
            use_layernorm = False
            torch_model = RNN(input_size, hidden_size1, num_layers1, hidden_size2, num_layers2, num_classes, dropout_, bidirectional, bidirectional, pre_out, single_layer, linear_size, device, use_layernorm).to(device)
        if vars["model"] == "TCN":
            in_chans = 21 
            out_chans = 2 
            num_channels = [20, 20]
            kernel_size = 3
            dropout = .3
            torch_model = TCN(in_chans, out_chans, num_channels, kernel_size, dropout).to(device)
        if vars["model"] == "CNN":
            in_chans = 21
            out_chans =  [64]
            kernel_size = [2]
            stride = [1]
            dilation = [1]
            groups = [1]
            batch_norm = [True]
            dropout = [0.3]
            maxpool = [True]
            maxpool_kernel_size = [2]
            dense_out_sizes = [10]
            dense_pool = True
            dense_pool_kernel_size = 2
            dense_dropout = [0]
            global_average_pool = True
            num_classes = 2
            time_steps = strides_per_sequence
            position_encoding = True
            torch_model = CNN1D(in_chans, out_chans, kernel_size, stride, dilation, groups, batch_norm, dropout, maxpool, maxpool_kernel_size, dense_out_sizes, dense_pool, dense_pool_kernel_size, dense_dropout, global_average_pool, num_classes, time_steps, position_encoding).to(device)
        if vars["model"] == "MSResNet":
            in_chans = 21
            layers = [1, 1, 1, 1]
            num_classes = 2
            torch_model = MSResNet(in_chans, layers, num_classes)
        
        if vars["model"] == "MSResNetRaw":
            in_chans = 21
            layers = [1, 1, 1, 1]
            num_classes = 2
            torch_model = MSResNetRaw(in_chans, layers, num_classes)

        if vars["model"] == "ResNet":
            in_chans =  21 
            initial_conv_layer = True
            block_name = "basic_block"
            layers = [1, 2, 0, 0]
            kernel_size_conv1 =  8 
            kernel_size_conv2 = 5 
            kernel_size_conv3 = 3 
            stride_layer64 = [1, 1, 1] 
            stride_layer128 = [1, 1, 1] 
            stride_layer256 = [1, 1, 1] 
            stride_layer512 = [1, 1, 1]
            position_encoding = True
            num_classes = 2
            torch_model = ResNet(in_chans, initial_conv_layer, block_name, layers, kernel_size_conv1, kernel_size_conv2, kernel_size_conv3, stride_layer64, stride_layer128, stride_layer256, stride_layer512, position_encoding, num_classes)
            
        model = create_model(torch_model, device, bidirectional, lpath)
        start_time = time.time()
        grid_search = task_train(model, fullTrainLabelsList, trainStridesList_norm, params)
        cv_results = pd.DataFrame(grid_search.cv_results_)
        end_time = time.time()
        print("Time taken: ", end_time - start_time)
        #print(cv_results)
        cv_results.to_csv(lpath+"cv_results.csv")
        task_learning_curve(grid_search, lpath)
        print(grid_search.best_params_)
        #print(grid_search.best_estimator_.history)
        loc = "results/task_generalization_lstm/"+str(5)+"strides/"+time_now+dataset+framework+str(bidirectional) #Note: strides per sequence is hardcoded to 5
        #with open(loc+"history.json", 'wb') as f:
        #    grid_search.best_estimator_.history.to_file(f)
        model = grid_search #check if grid_search and grid_search.best_estimator_ are the same
        #save grid_search or model or best estimator?
        save_model(model.best_estimator_, lpath)
    if behavior == "evaluate":
        model = load_model(saved_model_path)
    if behavior == "feature_importance":
        task_gen_perm_imp_main_setup(data, vars["config_path"], vars["model_path"]) 
        exit()

    total_parameters = sum(p.numel() for p in model.best_estimator_.module.parameters())
    trainable_params =  sum(p.numel() for p in model.best_estimator_.module.parameters() if p.requires_grad)
    nontrainable_params = total_parameters - trainable_params

    #should evaluate be done on model, grid search, or best estimator
    testStridesList_norm = Variable(torch.tensor(np.array(testStridesList_norm))).float()
    eval_start_time = time.time()
    probs, results = evaluate_task_generalize_lstm(model, testStridesList_norm, fullTestLabelsList, lpath) #update to contain strides per sequence
    eval_end_time = time.time()
    print("Evaluation time: ", eval_end_time - eval_start_time)
    #Need some way to actually save results probably
    plotROC_task_generalize_LSTM(model, fullTestLabelsList, probs, results[4], lpath)
    print("results", results)
    path = results_path_task_generalize_lstm+str(strides_per_sequence)+"strides/"+time_now+dataset+str(bidirectional)+"results.csv"
    #acc, p, r, f1, auc, person_acc, person_p, person_r, person_f1, person_auc
    resultsdf = pd.DataFrame({'accuracy': results[0], 'precision': results[1], 'recall': results[2], 'f1': results[3], 'auc': results[4], 'person accuracy': results[5],\
        'person precision': results[6], 'person recall': results[7], 'person f1': results[8], 'person auc': results[9],\
	    'cross validation time': end_time - start_time, 'eval time':eval_end_time - eval_start_time,\
        'Model Parameters': total_parameters, 'Trainable Parameters': trainable_params, 'Nontrainable Parameters': nontrainable_params,\
        'Best Parameters': model.best_params_}, index=[0])
    resultsdf.to_csv(lpath+"results.csv")
    #np.savetxt(lpath+"best_params.txt", model.best_params_)

    #best_params = pd.DataFrame(model.best_params_, index=[0])
    try:
        best_model_params = pd.DataFrame(model.best_params_, index = [0])
    except:
        #best_params_dict["net__module__num_channels"] = str(best_params_dict["net__module__num_channels"])
        best_params_dict = {}
        for k, v in model.best_params_.items():
            if type(v) == list:
                best_params_dict[k] = str(v) 
        best_model_params = pd.DataFrame(best_params_dict, index=[0])
    best_model_params.to_csv(lpath+"best_params.csv")
    #Save total parameters and trainable parameters for final model to csv, also what the final model 
    #np.savetxt(path, results, delimiter =", ", fmt ='% s')
    #Add plotROC function


if framework == "subject":
    if "raw_features" not in vars["dataset"]:
        stridesList, labelsList = genSequences(data, strides_per_sequence, 2, vars["ablation"])
    else:
        trainStridesList, fullTrainLabelsList, testStridesList, fullTestLabelsList = read_raw_data("data/grouped_labels.csv", False, vars["dataset"])
        stridesList = trainStridesList + testStridesList
        #labelsList = fullTrainLabelsList + fullTestLabelsList
        print(type(fullTestLabelsList))
        labelsList = pd.concat([fullTrainLabelsList, fullTestLabelsList])
        labelsList.columns = ["Label", "PID"]
        print(np.array(stridesList).shape)
        print(np.array(labelsList).shape)
    #time_now = datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    if behavior == "train" or behavior == "feature_importance":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch_model = None
        if vars["model"] == "LSTM":
            input_size = 21
            num_classes = 2
            pre_out = 1
            torch_model = LSTM(input_size, hidden_size1, num_layers1, hidden_size2, num_layers2, num_classes, dropout_, bidirectional, bidirectional, pre_out, single_layer, linear_size, device, use_layernorm).to(device)
        if vars["model"] == "GRU":
            input_size = 21
            num_classes = 2
            pre_out = 1
            torch_model = GRU(input_size, hidden_size1, num_layers1, hidden_size2, num_layers2, num_classes, dropout_, bidirectional, bidirectional, pre_out, single_layer, linear_size, device, use_layernorm).to(device)
        if vars["model"] == "RNN":
            input_size = 21
            num_classes = 2
            pre_out = 1
            torch_model = RNN(input_size, hidden_size1, num_layers1, hidden_size2, num_layers2, num_classes, dropout_, bidirectional, bidirectional, pre_out, single_layer, linear_size, device, use_layernorm).to(device)
        if vars["model"] == "TCN":
            in_chans = 21 
            out_chans = 2 
            num_channels = [20, 20]
            kernel_size = 3
            dropout = .3
            torch_model = TCN(in_chans, out_chans, num_channels, kernel_size, dropout).to(device)
        if vars["model"] == "CNN":
            in_chans = 21
            out_chans =  [64]
            kernel_size = [2]
            stride = [1]
            dilation = [1]
            groups = [1]
            batch_norm = [True]
            dropout = [0.3]
            maxpool = [True]
            maxpool_kernel_size = [2]
            dense_out_sizes = [10]
            dense_pool = True
            dense_pool_kernel_size = 2
            dense_dropout = [0]
            global_average_pool = True
            num_classes = 2
            time_steps = strides_per_sequence
            position_encoding = True
            torch_model = CNN1D(in_chans, out_chans, kernel_size, stride, dilation, groups, batch_norm, dropout, maxpool, maxpool_kernel_size, dense_out_sizes, dense_pool, dense_pool_kernel_size, dense_dropout, global_average_pool, num_classes, time_steps, position_encoding).to(device)
        if vars["model"] == "MSResNet":
            in_chans = 21
            layers = [1, 1, 1, 1]
            num_classes = 2
            torch_model = MSResNet(in_chans, layers, num_classes)

        if vars["model"] == "MSResNetRaw":
            in_chans = 21
            layers = [1, 1, 1, 1]
            num_classes = 2
            torch_model = MSResNetRaw(in_chans, layers, num_classes)

        if vars["model"] == "ResNet":
            in_chans =  21
            initial_conv_layer = True
            block_name = "basic_block"
            layers = [1, 2, 0, 0]
            kernel_size_conv1 =  8 
            kernel_size_conv2 = 5 
            kernel_size_conv3 = 3 
            stride_layer64 = [1, 1, 1] 
            stride_layer128 = [1, 1, 1] 
            stride_layer256 = [1, 1, 1] 
            stride_layer512 = [1, 1, 1]
            position_encoding = True
            num_classes = 2
            torch_model = ResNet(in_chans, initial_conv_layer, block_name, layers, kernel_size_conv1, kernel_size_conv2, kernel_size_conv3, stride_layer64, stride_layer128, stride_layer256, stride_layer512, position_encoding, num_classes)

        print(lpath)
        if behavior != "feature_importance":
            model = create_model(torch_model, device, bidirectional, lpath)

        if behavior == "feature_importance":
            model = create_model(torch_model, device, bidirectional, None)
            subject_gen_perm_imp_main_setup(data, model, stridesList, labelsList, subject_params, vars["config_path"])
            exit()
        
        start_time = time.time()
        grid_search, yoriginal, ypredicted, labelsList_shuffled, pipe = subject_train(model, stridesList, labelsList, subject_params, strides_per_sequence, None, lpath) #check to make sure yoriginal and ypredicted make sense size wise, same length, only 0 or 1
        end_time = time.time()
        print("Time taken: ", end_time - start_time)
        #what to do with gridsearch?
        best_index = grid_search.cv_results_['mean_test_accuracy'].argmax()
        print(best_index)
        print('best_params: ', grid_search.cv_results_['params'][best_index])
        cv_results = pd.DataFrame(grid_search.cv_results_)
        print(cv_results)
        cv_results.to_csv(lpath + "cv_results.csv")
        #print("yoriginal", yoriginal)
        #print("ypredicted", ypredicted)
        #print(grid_search)
        #save_model(grid_search.net__best_estimator_, dataset, strides_per_sequence, framework, bidirectional, save_model_destination_subject)
        path = results_path_subject_generalize_lstm+str(strides_per_sequence)+"strides/"+time_now+dataset+str(bidirectional)
        #np.savetxt(path+"yoriginal.csv", yoriginal, delimiter =", ", fmt ='% s')
        #np.savetxt(path+"ypredicted.csv", ypredicted, delimiter =", ", fmt ='% s')
    if behavior == "evaluate":
        model = load_model(saved_model_path)
        yoriginal = np.genfromtxt(saved_model_path+"yoriginal.csv", delimiter=',')
        ypredicted = np.genfromtxt(saved_model_path+"ypredicted.csv", delimiter=',')

    #Not sure if this will work right
    #total_parameters = sum(p.numel() for p in model.best_estimator_.module.parameters())
    #trainable_params =  sum(p.numel() for p in model.best_estimator_.module.parameters() if p.requires_grad)
    #nontrainable_params = total_parameters - trainable_params

    #print("best estimator", grid_search.best_estimator_)

    learning_curves(grid_search, pipe, stridesList, labelsList, lpath)
    
    best_params_dict = grid_search.cv_results_['params'][best_index]

    #Get # of parameters

    best_params = grid_search.cv_results_['params'][best_index]
    total_parameters = sum(p.numel() for p in pipe.set_params(**best_params)['net'].module.parameters())     
    trainable_params =  sum(p.numel() for p in pipe.set_params(**best_params)['net'].module.parameters() if p.requires_grad)
    nontrainable_params = total_parameters - trainable_params



    '''
    net = NeuralNetClassifier(
        torch_model,
        max_epochs = 1,
        lr = .01,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.SGD,
        device = device,
        iterator_train__shuffle=True,
        batch_size=100,
        callbacks=[EarlyStopping(patience = 100, lower_is_better = True, threshold=0.0001), 
        #('lr_scheduler', LRScheduler(policy=torch.optim.lr_scheduler.StepLR, step_size = 500)),
        (EpochScoring(scoring=accuracy_score, lower_is_better = False, on_train = True, name = "train_acc"))]
        #callbacks=[EarlyStopping(patience = 100, lower_is_better = True, threshold=0.0001),
        #(EpochScoring(scoring=accuracy_score, lower_is_better = False, on_train = True, name = "train_acc"))]
    )
    
    pipe = Pipeline([('scale', torch_StandardScaler(vars["strides_per_sequence"])), ('net', net)])
    fixed_params = {}
    for k, v in best_params_dict.items():
        k = k.replace("net__", "")
        fixed_params[k] = [v]
    Y1 = labelsList.iloc[:,0].astype('int64') #Dropping the PID
    #print(type(Y1))
    X = Variable(torch.tensor(np.array(stridesList))).float()
    gs = GridSearchCV(net, param_grid=fixed_params, refit = True)
    gs.fit(X, Y1)

    total_parameters = sum(p.numel() for p in gs.best_estimator_.module.parameters())
    trainable_params =  sum(p.numel() for p in gs.best_estimator_.module.parameters() if p.requires_grad)
    nontrainable_params = total_parameters - trainable_params
    '''

    best_params_dict["total_parameters"] = total_parameters
    best_params_dict["trainable_parameters"] = trainable_params
    best_params_dict["nontrainable_params"] = nontrainable_params


    eval_start_time = time.time()
    probs, results = evaluate_subject_generalize_lstm(grid_search, labelsList_shuffled, yoriginal, ypredicted, lpath) #not sure what input to give this?, how to run evaluate for subject wise
    eval_end_time = time.time()
    path = results_path_subject_generalize_lstm+str(strides_per_sequence)+"strides/"+time_now+dataset+str(bidirectional)+"results.csv"
    
    resultsdf = pd.DataFrame(results, index =['Stride Mean', 'Stride SD', 'Person Mean', 'Person SD'], 
                                              columns =['accuracy', 'precision', 'recall', 'f1', 'auc'])
    resultsdf.to_csv(lpath+"results.csv")
    predicted_probs = pd.DataFrame({"Probs": ypredicted, "Label": yoriginal})
    predicted_probs.to_csv(lpath+"probs.csv")


    best_params_dict["train_time"] = end_time - start_time
    best_params_dict["eval_time"] = eval_end_time - eval_start_time

    print(best_params_dict)
    try:
        best_model_params = pd.DataFrame(best_params_dict, index=[0])
    except:
        #best_params_dict["net__module__num_channels"] = str(best_params_dict["net__module__num_channels"])
        for k, v in best_params_dict.items():
            if type(v) == list:
                best_params_dict[k] = str(v) 
        best_model_params = pd.DataFrame(best_params_dict, index=[0])
    best_model_params.to_csv(lpath+"best_params.csv")
    #print("results", results)
    #plot roc for subject generalize

    #make sure to save predicted probabilities and metrics to csv

