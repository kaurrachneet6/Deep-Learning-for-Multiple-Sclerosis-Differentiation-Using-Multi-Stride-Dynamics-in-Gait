from importlib import reload

from matplotlib.pyplot import grid
from torch.utils import data
import utils.package_imports
reload(utils.package_imports)
from utils.package_imports import *

seed = 5

#default lstm params:
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
dropout = .2
bi_mult1 = 1
bi_mult2 = 1
single_layer = True
linear_size = 1



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

def read_raw_data(data_path, is_task, datatype):
    files = pd.read_csv(data_path, index_col = 0)
    training = files[files['TrialID'] == 1]
    training_files = []
    training_labels = []
    if datatype == "sizeN_raw_features":
        path = "data/sizeN_grouped_5strides/"
    else:
        path = "data/grouped_5strides/"
    #print(training)
    for index, file in training.iterrows():
        #print(file)
        read_file = pd.read_csv(path+file["FileName"], index_col = 0)
        training_files.append(read_file.to_numpy())
        #print(file.loc[['PID', 'label']])
        labels = file.loc[['label', 'PID']].to_numpy().tolist()

        training_labels.append(labels)
    
    if not is_task:
        training_labels = training[['label', 'PID']]
    
    testing = files[files['TrialID'] == 2]
    testing_files = []
    testing_labels = []
    for index, file in testing.iterrows():
        read_file = pd.read_csv(path+file["FileName"], index_col = 0)
        testing_files.append(read_file.to_numpy())
        labels = file.loc[['label', 'PID']].to_numpy().tolist()
        testing_labels.append(labels)

    if not is_task:
        testing_labels = testing[['label', 'PID']]


    return training_files, training_labels, testing_files, testing_labels


def data_stream_features(data_stream):
    #spatial = ['stride_length', 'stride_width', 'LeftFPA', 'RightFPA']
    #temporal = ['stride_time', 'swing_time', 'stance_time', 'SS_R', 'DS_L', 'DS_R', 'cadence' ]
    #spatiotemporal = ['stride_speed', 'walk_ratio']
    #kinetic = ['force_HSR', 'force_MidSSR', 'force_TOR', 'force_HSL', 'force_TOL', 'force_MidSSL', 'Butterfly_x_abs', 'ButterflySQ_x']
    if data_stream == "spatial":
        return ['stride_length', 'stride_width', 'LeftFPA', 'RightFPA'] #4
    if data_stream == "temporal":
        return ['stride_time', 'swing_time', 'stance_time', 'SS_R', 'DS_L', 'DS_R', 'cadence'] #7
    if data_stream == "kinetic":
        return ['force_HSR', 'force_MidSSR', 'force_TOR', 'force_HSL', 'force_TOL', 'force_MidSSL', 'Butterfly_x_abs', 'ButterflySQ_x'] #8
    if data_stream == "spatiotemporal":
        return ['stride_speed', 'walk_ratio', 'stride_length', 'stride_width', 'LeftFPA', 'RightFPA', 'stride_time', 'swing_time', 'stance_time', 'SS_R', 'DS_L', 'DS_R', 'cadence'] #13
    if data_stream == "spatial_kinetic":
        return ['stride_length', 'stride_width', 'LeftFPA', 'RightFPA', 'force_HSR', 'force_MidSSR', 'force_TOR', 'force_HSL', 'force_TOL', 'force_MidSSL', 'Butterfly_x_abs', 'ButterflySQ_x'] #12
    if data_stream == "temporal_kinetic":
        return ['stride_time', 'swing_time', 'stance_time', 'SS_R', 'DS_L', 'DS_R', 'cadence', 'force_HSR', 'force_MidSSR', 'force_TOR', 'force_HSL', 'force_TOL', 'force_MidSSL', 'Butterfly_x_abs', 'ButterflySQ_x'] #15
    if data_stream == "wearable":
        return ['stride_length', 'stride_width', 'LeftFPA', 'RightFPA', 'stride_time', 'swing_time', 'stance_time', 'SS_R', 'DS_L', 'DS_R', 'cadence', 'stride_speed', 'walk_ratio', 'force_HSR', 'force_MidSSR', 'force_TOR', 'force_HSL', 'force_TOL', 'force_MidSSL']
    return ['stride_length', 'stride_width', 'LeftFPA', 'RightFPA', 'stride_time', 'swing_time', 'stance_time', 'SS_R', 'DS_L', 'DS_R', 'cadence', 'stride_speed', 'walk_ratio', 'force_HSR', 'force_MidSSR', 'force_TOR', 'force_HSL', 'force_TOL', 'force_MidSSL', 'Butterfly_x_abs', 'ButterflySQ_x']

def data_stream_num_features(data_stream):
    #spatial: 4
    #temporal: 7
    #kinetic: 8
    #spatiotemporal: 13
    #spatial_kinetic: 12
    #temporal_kinetic: 15
    #wearable: 19
    return len(data_stream_features(data_stream))

def getFeatureIndices(data, data_stream):
    rl = []
    data = data.drop(['PID', 'Label', 'TrialID', 'id'], axis = 1)
    for c in data_stream_features(data_stream):
        if c in data:
            rl.append(data.columns.get_loc(c))
    #return [data.columns.get_loc(c) for c in data_stream_features(data_stream) if c in data]
    return rl

def getIndividualFeatureIndex(data, feature):
    #print("feature: ", feature)
    data = data.drop(['PID', 'Label', 'TrialID', 'id'], axis = 1)
    #print(data.columns)
    #print([data.columns.get_loc(feature)])
    return [data.columns.get_loc(feature)]

def genTTSequences(data, strides_per_sequence, skippedSteps, data_stream = "all"):
    '''
    Function to generate sequences for task gen
    Arguments:
        data: Prepared dataframe of strides
        strides_per_sequence: amount of stides that will be in each generated sequence
        skippedSteps: Amount of steps to move after valid sequence is created
    Returns:
        Lists with testing and training labels and sequences
    '''
    row = 0
    nullseq = 0;

    id = data.loc[0, 'id']
    pid = data.loc[0, 'PID']
    testStridesList = []
    testLabelsList = []
    trainStridesList = []
    trainLabelsList = []
    fullTestLabelsList = []
    fullTrainLabelsList = []

    features = data_stream_features(data_stream)
    info = ['id', 'PID', 'Label', 'TrialID']
    features.extend(info)
    data = data[features]
    
    while row < (data.shape[0] - strides_per_sequence):
        if data.loc[row, 'id'] == id and data.loc[row+strides_per_sequence, 'id'] == id: #and data.loc[row, 'Label'] == 0: #This last part is for testing
            #seq = data.iloc[row:row+stridesPerSequence].drop(['PID', 'Label', 'TrialID', 'id'], axis = 1)
            seq = data.iloc[row:row+strides_per_sequence].drop(['PID', 'Label', 'id'], axis = 1)
            if not seq.isnull().values.any():
                if data.loc[row, 'TrialID'] == 2:
                    testLabelsList.append((data.loc[row, 'Label']))
                    fullTestLabelsList.append(data.loc[row, ['Label', 'PID']].to_numpy().tolist())
                    #keep pid number, as a second dimention only send in first dimention, sequence
                    testStridesList.append(data.iloc[row:row+strides_per_sequence].drop(['PID', 'Label', 'TrialID', 'id'], axis = 1).to_numpy())
                else:
                    trainLabelsList.append((data.loc[row, 'Label']))
                    fullTrainLabelsList.append(data.loc[row, ['Label', 'PID']].to_numpy().tolist())
                    trainStridesList.append(data.iloc[row:row+strides_per_sequence].drop(['PID', 'Label', 'TrialID', 'id'], axis = 1).to_numpy())
                row +=skippedSteps
            else:
                nullseq+=1
                row+=1
            
        else:
            id = data.loc[row, 'id']
            pid = data.loc[row,'PID']
            row+=1
    
    print("dropped sequences: ", nullseq)
    return testStridesList, fullTestLabelsList, trainStridesList, fullTrainLabelsList

def genSequences(data, strides_per_sequence, skippedSteps, data_stream = "all"):
    '''
    Function to generate sequences for task gen
    Arguments:
        data: Prepared dataframe of strides
        strides_per_sequence: amount of stides that will be in each generated sequence
        skippedSteps: Amount of steps to move after valid sequence is created
    Returns:
        Lists with testing and training labels and sequences
    '''
    row = 0
    nullseq = 0;

    id = data.loc[0, 'id']
    pid = data.loc[0, 'PID']
    stridesList = []
    labelsList = pd.DataFrame()

    #for ablation
    features = data_stream_features(data_stream)
    info = ['id', 'PID', 'Label', 'TrialID']
    features.extend(info)
    data = data[features]
    
    while row < (data.shape[0] - strides_per_sequence):
        if data.loc[row, 'id'] == id and data.loc[row+strides_per_sequence, 'id'] == id: #and data.loc[row, 'Label'] == 0: #This last part is for testing
            #seq = data.iloc[row:row+stridesPerSequence].drop(['PID', 'Label', 'TrialID', 'id'], axis = 1)
            seq = data.iloc[row:row+strides_per_sequence].drop(['PID', 'Label', 'id'], axis = 1)
            if not seq.isnull().values.any():
                #labelsList.append(data.loc[row, ['Label', 'PID']].to_numpy().tolist())
                labelsList = labelsList.append(data.loc[row, ['Label', 'PID']])
                stridesList.append(data.iloc[row:row+strides_per_sequence].drop(['PID', 'Label', 'TrialID', 'id'], axis = 1).to_numpy())
                row +=skippedSteps
            else:
                nullseq+=1
                row+=1
            
        else:
            id = data.loc[row, 'id']
            pid = data.loc[row,'PID']
            row+=1
    
    print("dropped sequences: ", nullseq)
    return stridesList, labelsList


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
    print("mean", mean)
    print("std", sd)
    return mean, sd

def set_random_seed(seed_value):
    '''
    To set the random seed for reproducibility of results
    Arguments: seed value and use cuda (True if cuda is available)
    '''
    random.seed(seed_value)
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    use_cuda = torch.cuda.is_available() #use_cuda is True if cuda is available
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    #torch.set_deterministic(True)
    if use_cuda:
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        

class MyCheckpoint(TrainEndCheckpoint):
	def on_train_begin(self, net, X, y, **kwargs):
		self.fn_prefix = 'train_end_' + str(id(X))


def create_model(model, device, bidirectional = False, lpath = None):
    '''
    Creates Skorch LSTM model
    Arguments:
        bidirectional: Whether the lstm is bidirectional
    Returns:
        Created skorch network
    '''
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = LSTM(input_size, hidden_size1, num_layers1, hidden_size2, num_layers2, num_classes, dropout, bidirectional, bidirectional, pre_out, single_lstm, linear_size).to(device)
    #set_random_seed(seed)
    net = NeuralNetClassifier(
        model,
        max_epochs = 10,
        lr = .01,
        criterion=nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        device = device,
        iterator_train__shuffle=True,
        batch_size=100,
        train_split = dataset.CVSplit(5, random_state = 0),
        callbacks=[EarlyStopping(patience = 10, lower_is_better = True, threshold=0.0001),
            (FixRandomSeed()),
        #('lr_scheduler', LRScheduler(policy=torch.optim.lr_scheduler.StepLR, step_size = 500)),
            (EpochScoring(scoring=accuracy_score, lower_is_better = False, on_train = True, name = "train_acc")),
            ((MyCheckpoint(f_params='params.pt', f_optimizer='optimizer.pt', f_criterion='criterion.pt', f_history='history.json', f_pickle=None, fn_prefix='train_end_', dirname= lpath)))] #add path
        #callbacks=[EarlyStopping(patience = 100, lower_is_better = True, threshold=0.0001),
        #(EpochScoring(scoring=accuracy_score, lower_is_better = False, on_train = True, name = "train_acc"))]
    )
    return net

def task_train(model, fullTrainLabelsList, trainStridesList_norm, params):
    '''
    Tunes and trains skorch model for task generalization
    Arguments:
        model: Skorch model
        fullTrainLabelsList: List of training data labels and PIDs
        trainStridesList_norm: Normlized list of training sequences
        params: List of hyperparameters to optimize across
    Returns:
        Trained and tuned grid search object
    '''
    fullTrainLabelsList, trainStridesList_norm = shuffle(fullTrainLabelsList, trainStridesList_norm, random_state = 0) #shuffle data first
    ftll = np.array(fullTrainLabelsList)
    trainY1 = Variable(torch.tensor(ftll[:,0])).long() #Dropping the PID
    trainX = Variable(torch.tensor(np.array(trainStridesList_norm))).float()
    print("trainX", trainX.shape)
    print("trainY1", trainY1.shape)
    groups_ = ftll[:,1] 
    print("groups_: ", groups_)

    gkf = GroupKFold(n_splits=5)

    grid_search = GridSearchCV(model, param_grid=params, scoring='accuracy', n_jobs = 1, \
                                    cv=gkf.split(trainX, trainY1, groups=groups_), refit = True, return_train_score = True)
    print("grid search", grid_search)

    for train, test in gkf.split(trainX, trainY1, groups=groups_):
        print("Cross val splits, class and PID")
        print('TRAIN: ', np.unique(ftll[train], axis=0), ' TEST: ', np.unique(ftll[test], axis=0))

    #Skorch callback history to get loss to plot
    grid_search.fit(trainX, trainY1, groups=groups_)
    return grid_search

def acc(y_true, y_pred):
    '''
    Returns the accuracy 
    Saves the true and predicted labels for training and test sets
    '''
    global yoriginal, ypredicted
    #yoriginal = []
    #ypredicted = []
    yoriginal.append(y_true)
    ypredicted.append(y_pred)
    #print("yoriginal acc:", yoriginal)
    #print("ypredicted acc:", ypredicted)
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def subject_train(model, stridesList, labelsList, params, stridesPerSequence, feature_importance = None, lpath = None):
    stridesList, labelsList = shuffle(stridesList, labelsList, random_state = 0) #shuffle data first
    #ftll = np.array(labelsList)
    #Y1 = Variable(torch.tensor(ftll[:,0])).long()##Dropping the PID
    print("labelslist 0: ", labelsList)
    Y1 = labelsList.iloc[:,0].astype('int64') #Dropping the PID
    #print(type(Y1))
    X = Variable(torch.tensor(np.array(stridesList))).float()
    print("Xshape: ", X.shape)
    #print("Y",Y1[0:5])
    #print("x",X[0:5])
    #print("x shape", np.array(X).shape)
    #lstm_grid = make_pipeline(custom_StandardScaler(stridesPerSequence), model)
    if feature_importance == None:
        pipe = Pipeline([('scale', torch_StandardScaler(stridesPerSequence, X.shape[2])), ('net', model)])
    else:
        #feature imporatance = feature indices
        pipe = Pipeline([('scale', torch_StandardScaler(stridesPerSequence, X.shape[2])), ('permutation', PermuteTransform(feature_importance)), ('net', model)])
    #pipe = Pipeline([('net', model),])
    #groups_ = ftll[:,1]
    groups_ = labelsList.iloc[:,1]
    gkf = GroupKFold(n_splits=5)

    print("Cross val split PIDs:\n")
    print(stridesPerSequence)
    if lpath != None and stridesPerSequence != 150:
        for idx, (train, test) in enumerate(gkf.split(X, Y1, groups=groups_)):
            pd.DataFrame(train).to_csv(lpath+"indices_train_fold_"+str(idx+1)+".csv")           #add folder path
            pd.DataFrame(test).to_csv(lpath+"indices_test_fold_"+str(idx+1)+".csv")
            pd.DataFrame(X[train]).to_csv(lpath+"indices_X_train_fold_"+str(idx+1)+".csv")           #add folder path
            pd.DataFrame(X[test]).to_csv(lpath+"indices_X_test_fold_"+str(idx+1)+".csv")
            pd.DataFrame(Y1.iloc[train]).to_csv(lpath+"indices_Y1_train_fold_"+str(idx+1)+".csv")           #add folder path
            pd.DataFrame(Y1.iloc[test]).to_csv(lpath+"indices_Y1_test_fold_"+str(idx+1)+".csv")
            #pd.DataFrame(Y1).to_csv(lpath+"indices_Y1_"+str(idx+1)+".csv")
            pd.DataFrame(groups_.iloc[train]).to_csv(lpath+"indices_groups_train_fold_"+str(idx+1)+".csv")           #add folder path
            pd.DataFrame(groups_.iloc[test]).to_csv(lpath+"indices_groups_test_fold_"+str(idx+1)+".csv")
            #pd.DataFrame(groups_).to_csv(lpath+"indices_groups_"+str(idx+1)+".csv")

    global yoriginal, ypredicted
    yoriginal = []
    ypredicted = []
    #print("yoriginal:", yoriginal)
    #print("ypredicted:", ypredicted)

    scores={'accuracy': make_scorer(acc), 'precision':'precision', 'recall':'recall', 'f1': 'f1', 'auc': 'roc_auc'}
    grid_search = GridSearchCV(pipe, param_grid=params, scoring=scores, n_jobs = 1, \
                                    cv=gkf.split(X, Y1, groups=groups_), refit = False) 
    #print("grid search", grid_search)
    #print("X", X)
    #print("Y1", Y1)
    grid_search.fit(X, Y1, groups=groups_)
    #model.fit(X, Y1)
    return grid_search, yoriginal, ypredicted, labelsList, pipe
    




def load_model(path):
    '''
    Loads a saved skorch model
    Arguments:
        path: file path to saved skorch model
    Returns:
        Loaded skorch model
    '''
    '''
    net = NeuralNetClassifier(
    module=LSTM,
    criterion=nn.CrossEntropyLoss)

    net.initialize()
    net.load_params(f_params=path+"model.pkl", f_optimizer=path+"opt.pkl", f_history=path+'history.json')
    return net
    '''
    with open(path+'.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def save_model(model, lpath):
    '''
    Saves a skorch model
    Arguments:
        model: Skorch model
        dataset: Type of data used
        strides_per_sequence: stides per sequence in data
        framework: task or subject framework
        bidirectional: is LSTM bidirectional
        save_model_destination: path to save model
    '''
    
    #loc = save_model_destination+str(strides_per_sequence)+"strides/"+time+dataset+framework+str(bidirectional)
    #model.save_params(f_params=loc+'model.pkl', f_optimizer=loc+'opt.pkl', f_history=loc+'history.json')
    
    with open(lpath+"model.pkl", 'wb') as f:
        pickle.dump(model, f)


#Define model
#modified from: github.com/AladdinPerzon/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_rnn_gru_lstm.py
"""
# Hyperparameters
input_size = 21 #We have 21 features 
hidden_size1 = 30
num_layers1 = 10
hidden_size2 = 30
num_layers2 = 10
bidirectional1 = True
bidirectional2 = True
pre_out= 50
num_classes = 2
sequence_length = 10
learning_rate = 0.0001
batch_size = 100
num_epochs = 10
dropout = .2
bi_mult1 = 1
bi_mult2 = 1
torch.manual_seed(seed)
"""

#dimentions among other things are not correct
class LSTM(nn.Module):
    '''
    Pytorch LSTM model class
    Functions:
        init: initializes model based on given parameters
        forward: forward step through model
    '''
    def __init__(self, input_size, hidden_size1, num_layers1, hidden_size2, num_layers2, num_classes, dropout, bidirectional1, bidirectional2, pre_out, single_lstm, linear_size):
        super(LSTM, self).__init__()
        self.hidden_size1 = hidden_size1
        self.num_layers1 = num_layers1
        self.input_size = input_size
        self.hidden_size2 = hidden_size2
        self.num_layers2 = num_layers2
        self.num_classes = num_classes
        self.dropout = dropout
        self.bidirectional1 = bidirectional1
        self.bidirectional2 = bidirectional2
        self.pre_out = pre_out
        self.single_lstm = single_lstm
        self.linear_size = linear_size
        self.batch_size = 100
        self.h0, self.c0, self.h02, self.c02 = self.init_hidden()
        #self.bi_mult1 = 1
        #self.bi_mult2 = 1
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        #Try splitting lstm into multiple with different hidden sizes 
        
        if (self.bidirectional1):
            bi_mult1 = 2
        else:
            bi_mult1 = 1      

        if (self.bidirectional2):
            bi_mult2 = 2
        else:
            bi_mult2 = 1
        
        if (self.single_lstm):
            hidden_size = hidden_size1
        else:
            hidden_size = hidden_size2

        self.prefc = nn.Linear(input_size, pre_out)
        #Does num_layers actually make multiple lstm layers?
        #self.lstm = nn.LSTM(pre_out, hidden_size1, num_layers1, batch_first=True, bidirectional = bidirectional1) #uncomment for first linear layer

        ##Use these two for two seperate lstm layers
        self.lstm = nn.LSTM(input_size, hidden_size1, num_layers1, batch_first=True, bidirectional = bidirectional1) #uncomment for first lstm layer

        #self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional = bidirectional) 
        self.lstm2 = nn.LSTM(hidden_size1*bi_mult1, hidden_size2, num_layers2, batch_first=True, bidirectional = bidirectional2)

        #Batch_first means shape is (batch, seq, feature)
        self.dropout = nn.Dropout(p=dropout)
        #self.fc = nn.Linear(hidden_size * sequence_length, num_classes)
        #self.fc = nn.Linear(hidden_size*2, 1) replaced with below

        self.fc0 = nn.Linear(hidden_size*bi_mult2, linear_size)

        #self.fc = nn.Linear(linear_size, 2)

        self.fc = nn.Linear(hidden_size*bi_mult2, 2)



        #print("fc constructor: ",hidden_size2*bi_mult2)
        #self.fc2 = nn.Linear(sequence_length, 2)
        self.softmax = nn.Softmax(dim=1)
    
    def init_hidden(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h = torch.empty(self.num_layers1*bi_mult1, batch_size, self.hidden_size1)
        c = torch.empty(self.num_layers1*bi_mult1, batch_size, self.hidden_size1)
        
        h0 = torch.nn.init.xavier_normal_(h, gain=1.0).to(device)
        c0 = torch.nn.init.xavier_normal_(c, gain=1.0).to(device)
        
        #h0 = torch.nn.init.kaiming_uniform_(h, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)
        #c0 = torch.nn.init.kaiming_uniform_(c, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)

        h2 = torch.empty(self.num_layers2*bi_mult2, batch_size, self.hidden_size2)
        c2 = torch.empty(self.num_layers2*bi_mult2, batch_size, self.hidden_size2)
        
        h02 = torch.nn.init.xavier_normal_(h2, gain=1.0).to(device)
        c02 = torch.nn.init.xavier_normal_(c2, gain=1.0).to(device)
        
        #h02 = torch.nn.init.kaiming_uniform_(h2, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)
        #c02 = torch.nn.init.kaiming_uniform_(c2, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)
        return h0, c0, h02, c02



    def forward(self, x):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(x.shape)
        #print("x.size(0): ", x.size(0))
        #print(x)
        #print(x.type())
        #print(x[0].type())
        #print(x[0][0].type())


        
        #set_random_seed(seed)
        
        # Set initial hidden and cell states
        #Use xavier initializtion instead of zeros
        #Or something else fancier: kaiming he
        if (self.bidirectional1):
            bi_mult1 = 2
        else:
            bi_mult1 = 1        

        if (self.bidirectional2):
            bi_mult2 = 2
        else:
            bi_mult2 = 1
        
        #Old initialization:
        '''
        h = torch.empty(self.num_layers1*bi_mult1, x.size(0), self.hidden_size1)
        c = torch.empty(self.num_layers1*bi_mult1, x.size(0), self.hidden_size1)
        
        h0 = torch.nn.init.xavier_normal_(h, gain=1.0).to(device)
        c0 = torch.nn.init.xavier_normal_(c, gain=1.0).to(device)
        
        #h0 = torch.nn.init.kaiming_uniform_(h, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)
        #c0 = torch.nn.init.kaiming_uniform_(c, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)

        h2 = torch.empty(self.num_layers2*bi_mult2, x.size(0), self.hidden_size2)
        c2 = torch.empty(self.num_layers2*bi_mult2, x.size(0), self.hidden_size2)
        
        h02 = torch.nn.init.xavier_normal_(h2, gain=1.0).to(device)
        c02 = torch.nn.init.xavier_normal_(c2, gain=1.0).to(device)
        
        #h02 = torch.nn.init.kaiming_uniform_(h2, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)
        #c02 = torch.nn.init.kaiming_uniform_(c2, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)
        '''
        
        #Initialize initial LSTM states
        h0 = self.h0
        c0 = self.c0
        h02 = self.h02
        c02 = self.c02
        if x.size(0) != self.batch_size:
            h0, c0, h02, c02 = self.init_hidden(x.size(0))
        
        #h02 = torch.nn.init.kaiming_uniform_(h2, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)
        #c02 = torch.nn.init.kaiming_uniform_(c2, a=0, mode='fan_in', nonlinearity='leaky_relu').to(device)

        #linear layer takes features in and then output as different dimention, change LSTM size to match
        #print("x shape: ", x.shape)

        #out = F.relu(self.prefc(x)) #Uncomment for first linear layer
        #out = (self.prefc(x))
        #print("prefc shape: ", out.shape)

        # Forward propagate LSTM
        #, _ = self.lstm(out, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size) #uncomment for first linear layer
        out, _ = self.lstm(x, (h0, c0))    #uncomment for first lstm layer
        #print("lstm1 out: ", out.shape)
        if (not self.single_lstm):
            out, _ = self.lstm2(out, (h02, c02))
        #print("lstm2 out: ", out.shape)
        #out = out.reshape(out.shape[0], -1)
        #print(torch.isnan(out).any())
        #print("lstm out shape: ", out.shape)
        out = out[:,-1,:]
        #print("Last stride: ", out.shape)
        #print("fixed shape: ", out.shape)
        out = self.dropout(out)
        #print("dropout shape: ", out.shape)
        #print(out)
        #print("after dropout: ", out.shape)
        #print("fc expected size: ", hidden_size2*bi_mult2)
        #out = self.fc(out[:,:-1])
        #out = F.tanh(self.fc0(out))
        out = F.tanh(self.fc(out))
        #print("fully connected 1 shape: ", out.shape)
        #print(out)
        
        #Look at flattening instead of squeeze to reduce 2 fc layers to one
        #out = torch.squeeze(out)
        #out = F.relu(self.fc2(out))
        #print("fully connected 2 shpae: ", out.shape)
        
        #out = self.softmax(out)
        #print("softmax out shape: ", out.shape)
        #print(out)
        
        

        #check: is lstm output 10x30x2? if so it is all timesteps output, we only want the last one out[:,:,-1]? shape should be 30x2 for last one

        
        return out


def evaluate_task_generalize_lstm(model, test_features, trueY, lpath):
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
    Returns:
        proportion_strides_correct['prob_class1']: Prediction probabilities for HOA/MS for the ROC curve
        [acc, p, r, f1, auc, person_acc, person_p, person_r, person_f1, person_auc]: sequence and subject wise 
                                                   evaluation metrics (Accuracy, Precision, Recall, F1 and AUC)
    '''
    trueY = np.array(trueY)
    trueY = pd.DataFrame({'label': trueY[:, 0], 'PID': trueY[:, 1]})
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
    plt.savefig(lpath+'CFmatrix_task_generalize_seq_wise.png', dpi = 350) #Save to strides specfic folder
    #plt.show()
    
    #Subject wise confusion matrix
    plt.figure()
    confusion_matrix = pd.crosstab(proportion_strides_correct['True Label'], proportion_strides_correct['Predicted Label'], \
                                   rownames=['Actual'], colnames=['Predicted'], margins = True)
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", fmt = 'd')
    plt.savefig(lpath+'CFmatrix_task_generalize_subject_wise.png', dpi = 350)
    #plt.show()
    
    return proportion_strides_correct['prob_class1'], [acc, p, r, f1, auc, person_acc, person_p, person_r, person_f1, person_auc]

#Not sure what input this takes exactly
def evaluate_subject_generalize_lstm(model, Y, yoriginal_, ypredicted_, lpath):
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
    Returns: 
        tpr_list: True positive rate for the ROC curve 
        [stride_metrics_mean, stride_metrics_std, person_means, person_stds]: Means and standard deviations for the 
                                                                        sequence and subject based evaluation metrics 
    '''      
    trueY = Y
    #trueY = np.array(Y)
    #rueY = pd.DataFrame({'label': trueY[:, 0], 'PID': trueY[:, 1]})
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
    n_folds = 5 #THIS SHOULD BE A PARAMETER SENT IN 
    person_acc, person_p, person_r, person_f1, person_auc = [], [], [], [], []
    #For ROC curves 
    tpr_list = []
    base_fpr = np.linspace(0, 1, 101)

    for i in range(n_folds):
        #For each fold, there are 2 splits: test and train (in order) and we need to retrieve the index 
        #of only test set for required 5 folds (best index)
        #print("trueY:", trueY)
        #print("yoriginal:", yoriginal_[(best_index*n_folds)+ (i)])
        temp = trueY.loc[yoriginal_[(best_index*n_folds) + (i)].index] #True labels for the test strides in each fold
        temp['pred'] = ypredicted_[(best_index*n_folds) + (i)] #Predicted labels for the strides in the test set in each fold

        #Correctly classified strides i.e. 1 if stride is correctly classified and 0 if otherwise
        temp['correct'] = (temp['Label']==temp['pred'])
        
        #Appending the test strides' true and predicted label for each fold to compute stride-wise confusion matrix 
        test_strides_true_predicted_labels = test_strides_true_predicted_labels.append(temp)
        
        #Proportion of correctly classified strides
        proportion_strides_correct = temp.groupby('PID').aggregate({'correct': 'mean'})  

        proportion_strides_correct['True Label'] = temp[['PID', 'Label']].groupby('PID').first() 

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

    test_subjects_true_predicted_labels.to_csv(lpath+"test_subject_true_predicted_labels.csv")

    #Plotting and saving the sequence and subject wise confusion matrices 
    #Sequence wise confusion matrix
    plt.figure()
    confusion_matrix = pd.crosstab(test_strides_true_predicted_labels['Label'], test_strides_true_predicted_labels['pred'], \
                                   rownames=['Actual'], colnames=['Predicted'], margins = True)
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", fmt = 'd')
    plt.savefig(lpath + 'CFmatrix_subject_generalize_seq_wise.png', dpi = 350)
    #plt.show()

    #Subject wise confusion matrix
    plt.figure()
    confusion_matrix = pd.crosstab(test_subjects_true_predicted_labels['True Label'], test_subjects_true_predicted_labels['Predicted Label'], \
                                   rownames=['Actual'], colnames=['Predicted'], margins = True)
    sns.heatmap(confusion_matrix, annot=True, cmap="YlGnBu", fmt = 'd')
    plt.savefig(lpath + 'CFmatrix_subject_generalize_subject_wise.png', dpi = 350)
    #plt.ioff()
    #plt.show()
    #plt.close()
    return tpr_list, [stride_metrics_mean, stride_metrics_std, person_means, person_stds]





def plotROC_task_generalize_LSTM(model, test_Y, predicted_probs_person, metrics_personAUC, lpath):
    '''
    Function to plot and save the ROC curve for models given in ml_models list to the results directory
    Arguments: 
        ml_models: name of models to plot the ROC for 
        test_Y: true test set labels with PID
        predicted_probs_person: predicted test set probabilities
        metrics_personAUC: Subject-wise AUC for the corresponding data type (raw/size-N/regress-N) to label the ROC plot
        data_type: raw/sizeN/regressN_data to plot and save the ROC for
    '''
    ml_model_names = {'random_forest': 'RF', 'adaboost': 'Adaboost', 'kernel_svm': 'RBF SVM', 'gbm': 'GBM', \
                      'xgboost': 'Xgboost', 'knn': 'KNN', 'decision_tree': 'DT',  'linear_svm': 'LSVM', 
                 'logistic_regression': 'LR', 'mlp': 'MLP'}
    data_names = {'raw_data': 'Raw data', 'sizeN_data': 'Size-N data', 'regressN_data': 'Regress-N data'}
    test_Y = np.array(test_Y)
    test_Y = pd.DataFrame({'label': test_Y[:, 0], 'PID': test_Y[:, 1]})
    person_true_labels = test_Y.groupby('PID').first()
    neutral = [0 for _ in range(len(person_true_labels))] # ROC for majority class prediction all the time 

    fig, axes = plt.subplots(1, 1, sharex=True, sharey = True, figsize=(5.2, 3.5))
    sns.despine(offset=0)
    neutral_fpr, neutral_tpr, _ = roc_curve(person_true_labels, neutral) #roc curves
    linestyles = '-'
    colors = 'b'

    axes.plot(neutral_fpr, neutral_tpr, linestyle='--', label='Majority (AUC = 0.5)', linewidth = 3, color = 'k')
    
    model_probs = predicted_probs_person # person-based prediction probabilities
    fpr, tpr, _ = roc_curve(person_true_labels, model_probs)
    axes.plot(fpr, tpr, label="LSTM"+' (AUC = '+ str(round(metrics_personAUC, 3))
        +')', linewidth = 3, alpha = 0.8, linestyle = linestyles, color = colors)

    axes.set_ylabel('True Positive Rate')
    axes.set_title('Task generalization')
    plt.legend()
    # axes[1].legend(loc='upper center', bbox_to_anchor=(1.27, 1), ncol=1)

    axes.set_xlabel('False Positive Rate')
    plt.tight_layout()

    
    plt.savefig(lpath + 'ROC_task_generalize.png', dpi = 250)
    #plt.show()


#ROC curves for subject generalization framework
def plot_ROC_subject_generalize(model, tprs_list, metrics_personAUC, data_type = 'raw_data', results_dir = '5strides/'):
    '''
    Function to plot and save the ROC curve for subject generalization model given in ml_models list to the results directory
    Input: 
        ml_models: name of models to plot the ROC for 
        tprs_list: tprs list for all models of interest and the data type of interest
        metrics_personAUC: Subject-wise AUC for the corresponding data type (raw/size-N/regress-N) to label the ROC plot
        data_type: raw/sizeN/regressN_data to plot and save the ROC for
        results_dir: Depends on how many strides per sequence were selected for computing summary statistics
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

    tprs = tprs_list # person-based prediction probabilities
    tprs = np.array(tprs)
    mean_tprs = tprs.mean(axis=0)
    std = tprs.std(axis=0)

    tprs_upper = np.minimum(mean_tprs + std, 1)
    tprs_lower = mean_tprs - std
    #     axes[2].fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
    axes.plot(base_fpr, mean_tprs, label="LSTM"+' (AUC = '+ str(round(metrics_personAUC.loc['person_mean_AUC']
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
    plt.savefig(results_path_subject_generalize_trad + results_dir + 'ROC_subject_generalize_' + savefig_name + '.png', dpi = 250)
    #plt.show()



def task_learning_curve(model, lpath):
    epochs = [i for i in range(1, len(model.best_estimator_.history))] #start from 1 instead of zero
    train_loss = model.best_estimator_.history[:,'train_loss']
    valid_loss = model.best_estimator_.history[:,'valid_loss']
    train_acc = model.best_estimator_.history[:,'train_acc']
    valid_acc = model.best_estimator_.history[:,'valid_acc']
    #print("epochs", epochs, len(epochs))
    #print("train_acc", train_acc, len(train_acc))
    #print("train_loss", train_loss, len(train_loss))x
    #print("valid_loss", valid_loss, len(valid_loss))

    plt.plot(epochs,np.clip(train_loss[:-1], 0, 1),'g-'); #Dont print the last one for 3 built in
    plt.plot(epochs,np.clip(valid_loss[:-1], 0, 1),'r-');
    try:
        plt.plot(epochs, np.clip(train_acc, 0, 1),'b-');
    except:
        plt.plot(epochs, np.clip(train_acc[:-1], 0, 1),'b-');
    #plt.plot(np.arange(len(train_acc)),train_acc, 'b-'); #epochs and train_acc are off by one
    plt.plot(epochs,valid_acc[:-1], 'm-');
    plt.title('Training Loss Curves');
    plt.xlabel('Epochs');
    plt.ylabel('Mean Squared Error');
    plt.legend(['Train loss','Validation loss', 'Train Accuracy', 'Validation Accuracy']); 
    plt.savefig(lpath + 'learning_curve', dpi = 350)
    plt.show()

class custom_StandardScaler():
    def __init__(self, stridePerSequence, num_features):
        self.scaler = StandardScaler()
        self.stridesPerSequence = stridePerSequence
        self.num_features = num_features
    def fit(self,X,y=None):
        print(type(X))
        #Reshape
        Xr = X.reshape(-1, self.num_features)
        #Delete repeated rows
        Xu = np.unique(Xr, axis=0)
        self.scaler.fit(Xu)
        return self
    def transform(self,X,y=None):
        #Reshape it back to 3D and return 
        Xr = X.reshape(-1, self.num_features)
        X_new=self.scaler.transform(Xr)
        X_new = X_new.reshape(-1, self.stridesPerSequence, self.num_features)
        return X_new

class torch_StandardScaler():
    def __init__(self, stridePerSequence, num_features):
        self.scaler = StandardScaler()
        self.stridesPerSequence = stridePerSequence
        self.num_features = num_features
    def fit(self,X,y=None):
        X = X.numpy()
        #Reshape
        Xr = X.reshape(-1, self.num_features)
        #Delete repeated rows
        Xu = np.unique(Xr, axis=0)
        self.scaler.fit(Xu)
        return self
    def transform(self,X,y=None):
        #Reshape it back to 3D and return 
        X = X.numpy()
        Xr = X.reshape(-1, self.num_features)
        X_new=self.scaler.transform(Xr)
        X_new = X_new.reshape(-1, self.stridesPerSequence, self.num_features)
        return torch.from_numpy(X_new)

class FixRandomSeed(Callback):
    def __init__(self, seed=0):
        self.seed = 0
    def initialize(self):
        print("setting random seed to: ",self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        try:
            random.seed(self.seed)
        except NameError:
            import random
            random.seed(self.seed)
        np.random.seed(self.seed)
        torch.backends.cudnn.deterministic=True


#Learning curves     
def learning_curves(grid_search, pipe, stridesList, labelsList, path):
    '''
    To plot the training/validation loss and accuracy (stride-wise) curves over epochs across the n_splits folds 
    '''

    stridesList, labelsList = shuffle(stridesList, labelsList, random_state = 0)

    Y1 = labelsList.iloc[:,0].astype('int64') #Dropping the PID
    X = Variable(torch.tensor(np.array(stridesList))).float()
    groups_ = labelsList.iloc[:,1]
    gkf = GroupKFold(n_splits=5) 
    Y = Y1.values


    best_index = grid_search.cv_results_['mean_test_accuracy'].argmax()
    best_params = grid_search.cv_results_['params'][best_index]
    pipe_optimized = pipe.set_params(**best_params)
    #List of history dataframes over n_splits folds 
    histories = []     
    for fold, (train_ix, val_ix) in enumerate(gkf.split(X, Y1, groups=groups_)):
        # select rows for train and test
        #print("train_ix: ", train_ix)
        #print("X ", X[train_ix])
        #print("Y ", Y[train_ix])
        trainX, trainY, valX, valY = X[train_ix], Y[train_ix], X[val_ix], Y[val_ix]
        # fit model
        pipe_optimized.fit(trainX, trainY)
        history_fold = pd.DataFrame(pipe_optimized['net'].history)
#             print ('History for fold', fold+1, '\n')
#             display(history_fold)
        
        history_fold.to_csv(path + 'history_fold_' + str(fold+1) + '.csv')
        histories.append(history_fold)
        for idx in range(len(histories)):
            model_history = histories[idx]
            epochs = model_history['epoch'].values #start from 1 instead of zero
            train_loss = model_history['train_loss'].values
    #         print (train_loss)
            valid_loss = model_history['valid_loss'].values
            train_acc = model_history['train_acc'].values
            valid_acc = model_history['valid_acc'].values
            #print("epochs", epochs, len(epochs))
            #print("train_acc", train_acc, len(train_acc))
            #print("train_loss", train_loss, len(train_loss))
            #print("valid_loss", valid_loss, len(valid_loss))
            plt.plot(epochs,np.clip(train_loss, 0, 1),'g--'); #Dont print the last one for 3 built in
            plt.plot(epochs,np.clip(valid_loss, 0, 1),'r-');
            try:
                plt.plot(epochs,np.clip(train_acc, 0, 1),'b--');
            except:
                plt.plot(epochs,np.clip(train_acc[:-1], 0, 1),'b-');
            #plt.plot(np.arange(len(train_acc)),train_acc, 'b-'); #epochs and train_acc are off by one
            plt.plot(epochs,np.clip(valid_acc, 0, 10), 'm-');
    plt.title('Training/Validation loss and accuracy Curves');
    plt.xlabel('Epochs');
    plt.ylabel('Cross entropy loss/Accuracy');
    plt.legend(['Train loss','Validation loss', 'Train Accuracy', 'Validation Accuracy']); 
    
    plt.savefig(path + 'learning_curve', dpi = 350)
    plt.show()
    plt.close()


'''
Permutation Feature Importance 
'''
def task_gen_perm_imp_initial_setup(data, config_path, model_path):
    '''
    Permutation feature importance for task generalization initial setup
    '''
    #Task generalization W-> WT framework 

    testStridesList, fullTestLabelsList, trainStridesList, fullTrainLabelsList = genTTSequences(data, 5, 2, "all") #make sure to pass in data
    ss = custom_StandardScaler(5, np.array(testStridesList).shape[2])
    ss.fit(np.array(testStridesList))
    testStridesList_norm = ss.transform(np.array(testStridesList))
    trainStridesList_norm = ss.transform(np.array(trainStridesList))


    time_now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")
    path = os.getcwd()
    rpath = "results/feature_importance/"


    lpath = rpath+str(5)+"strides/Task/"+time_now+"/"
    print("lpath: ", lpath)
    gpath = path+"/"+lpath
    os.mkdir(gpath)

    with open(config_path, 'rb') as src, open(lpath+"/config.json", 'wb') as dst: dst.write(src.read())



    best_model = load_model(model_path) #change to passed arg

    testStridesList_norm = Variable(torch.tensor(np.array(testStridesList_norm))).float()
    print("input shape:", testStridesList_norm.shape)
    probs, results = evaluate_task_generalize_lstm(best_model, testStridesList_norm, fullTestLabelsList, lpath)
    resultsdf = pd.DataFrame({'accuracy': results[0], 'precision': results[1], 'recall': results[2], 'f1': results[3], 'auc': results[4], 'person accuracy': results[5],\
        'person precision': results[6], 'person recall': results[7], 'person f1': results[8], 'person auc': results[9]}, index=[0])
    perm_imp_results_df = pd.DataFrame(index = resultsdf.columns)

    return testStridesList_norm, best_model, lpath, perm_imp_results_df, fullTestLabelsList

def permute_shuffle(x, feature_indices):
    '''
    Each element in the testing set has body coords for features of interest replaced with the shuffled version 
    '''

    np.random.seed(random.randint(0, 100))
    shuffled = copy.deepcopy(x)
    print("shuffled shape: ", shuffled.shape)
    print("feature indices: ", feature_indices)
    for feat_index in feature_indices:
        np.random.shuffle(shuffled[:,:,feat_index])
    return shuffled


def task_gen_perm_imp_single_feature(feature, testStridesList_norm, best_model, fullTestLabelsList, lpath, perm_imp_results_df, data):
    '''
    Running the permutation feature importance for a single feature(group) say, left big toe
    Reference: https://christophm.github.io/interpretable-ml-book/feature-importance.html#fn35
    5 times, randomly permute the features of interest and with the newly set X_test, run the .predict and evaluate (with the already trained best model [The best model is read and defined in self.task_gen_perm_imp_initial_setup()]). Collect the metrics for the 5 runs, and compute the mean and standard deviation. 
    These metrics' mean and SD represent accuracy deleting the specific feature group, and thus lower the value than the original metric, more was the importance of the feature.
    Save csv files for 5+2 columns (original 5 runs + mean + SD) and no. of evaluation metrics rows for each feature and return the mean and SD to be collected by the main setup function to make a final csv with mean and SD metrics for all feature groups. 
    In this setup, we have 12 feature groups, namely left hip/knee/ankle/big toe/little toe/heel and similarly their right side counterparts. So this function should execute 12 times and return the corresponding mean and SD metrics. 
    '''
    #Column indices to permute in the X_sl_test_original to generate a new X_sl_test to predict and evaluate the trained model on 
    #self.feature_indices = self.testing_data.__define_column_indices_FI__(feature)
    #feature_indices = getFeatureIndices(data, feature)
    feature_index = getIndividualFeatureIndex(data, feature)
    #print (feature_indices)
    #Repeating the shuffling 5 times for randomness in permutations
    for idx in range(20):
        #Shuffling the features of interest
        shuffled = permute_shuffle(testStridesList_norm, feature_index)

        #Predicting the best trained model on shuffled data and computing the metrics 
        save_results_prefix = feature + '_' + str(idx)
        probs, results = evaluate_task_generalize_lstm(best_model, shuffled, fullTestLabelsList, lpath) #send in other stuff

        #Saving the metrics 
        perm_imp_results_df[save_results_prefix] = results #pass in
    feature_cols = [s for s in perm_imp_results_df.columns if feature in s]
    #Aggregating the mean and SD from the 5 random runs
    perm_imp_results_df[feature + '_' + 'mean'] = perm_imp_results_df[feature_cols].apply(pd.to_numeric, args=['coerce']).mean(axis=1, skipna=False)
    perm_imp_results_df[feature + '_' + 'std'] = perm_imp_results_df[feature_cols].apply(pd.to_numeric, args=['coerce']).std(axis=1, skipna=False)   

def task_gen_perm_imp_main_setup(data, config_path, model_path):
    '''
    Main setup for the permutation feature importance for task gen 
    Reference: https://christophm.github.io/interpretable-ml-book/feature-importance.html#fn35
    '''
    testStridesList_norm, best_model, lpath, perm_imp_results_df, fullTestLabelsList = task_gen_perm_imp_initial_setup(data, config_path, model_path)
    #12 Feature groups to explore the importance for 
    #features = ["all", "wearables", "kinetic", "spatial_kinetic", "spatial", "spatiotemporal", "temporal_kinetic", "temporal"]
    features = ['stride_length', 'stride_width', 'LeftFPA', 'RightFPA', 'stride_time', 'swing_time', 'stance_time', 'SS_R', 'DS_L',\
        'DS_R', 'cadence', 'stride_speed', 'walk_ratio', 'force_HSR', 'force_MidSSR', 'force_TOR', 'force_HSL', 'force_TOL',\
        'force_MidSSL', 'Butterfly_x_abs', 'ButterflySQ_x']
    for feature in features:
        #For all 12 feature groups 
        print ('Running for ', feature)
        task_gen_perm_imp_single_feature(feature, testStridesList_norm, best_model, fullTestLabelsList, lpath, perm_imp_results_df, data)
    display(perm_imp_results_df)
    #Saving all the 7*12 columns for all 12 feature groups and all 5 runs+2(mean/std)
    perm_imp_results_df.to_csv(lpath + 'Permutation_importance_all_results.csv')
    result_mean_cols = [s for s in  perm_imp_results_df.columns if 'mean' in s]
    result_std_cols = [s for s in  perm_imp_results_df.columns if 'std' in s]
    main_result_cols = result_mean_cols + result_std_cols
    #Saving only the mean and std per 12 feature groups (24 columns) that will be used to plot FI later
    perm_imp_results_df[main_result_cols].to_csv(lpath + 'Permutation_importance_only_main_results.csv')



'''
Permutation Feature Importance 
'''
def subject_gen_perm_imp_initial_setup(config_path):
    '''
    Permutation feature importance for subject generalization initial setup
    '''
    #Case when we can use all subjects in W or WT for full analysis 
    #Subject generalization W/WT framework 
    #Trial W/WT for training and testing both


    #stridesList, labelsList = genSequences(data, 5, 2, "all") #make sure to pass in data


    time_now = datetime.now().strftime("%Y_%m_%d-%H_%M_%S_%f")
    path = os.getcwd()
    rpath = "results/feature_importance/"


    lpath = rpath+str(5)+"strides/Subject/"+time_now+"/"
    gpath = path+"/"+lpath
    os.mkdir(gpath)

    with open(config_path, 'rb') as src, open(lpath+"/config.json", 'wb') as dst: dst.write(src.read())



    

    #testStridesList_norm = Variable(torch.tensor(np.array(testStridesList_norm))).float()
    #probs, results = evaluate_task_generalize_lstm(best_model, testStridesList_norm, fullTestLabelsList, lpath)

    #resultsdf = pd.DataFrame({'accuracy': results[0], 'precision': results[1], 'recall': results[2], 'f1': results[3], 'auc': results[4], 'person accuracy': results[5],\
    #    'person precision': results[6], 'person recall': results[7], 'person f1': results[8], 'person auc': results[9]}, index=[0])
    resultsdf = pd.DataFrame(columns =['Stride Mean accuracy', 'Stride Mean precision', 'Stride Mean recall', 'Stride Mean f1', 'Stride Mean auc',\
                                        'Stride SD accuracy',   'Stride SD precision',   'Stride SD recall',   'Stride SD f1',   'Stride SD auc',\
                                        'Person Mean accuracy', 'Person Mean precision', 'Person Mean recall', 'Person Mean f1', 'Person Mean auc',\
                                        'Person SD accuracy',   'Person SD precision',   'Person SD recall',   'Person SD f1',   'Person SD auc'])
    perm_imp_results_df = pd.DataFrame(index = resultsdf.columns)

    return lpath, perm_imp_results_df

def subject_gen_perm_imp_single_feature(feature, data, model, stridesList, labelsList, subject_params, perm_imp_results_df, lpath):   
    #Column indices to permute in the X_sl_test_original to generate a new X_sl_test to predict and evaluate the trained model on 
    
    feature_index = getIndividualFeatureIndex(data, feature)
    print (feature_index)
    #Repeating the shuffling 5 times for randomness in permutations
    for idx in range(20):
        grid_search, yoriginal, ypredicted, labelsList_shuffled, pipe = subject_train(model, stridesList, labelsList, subject_params, 5, feature_index)
        probs, results = evaluate_subject_generalize_lstm(grid_search, labelsList_shuffled, yoriginal, ypredicted, lpath)
        flat_list = [item for sublist in results for item in sublist]
        save_results_prefix = feature + '_' + str(idx)
        perm_imp_results_df[save_results_prefix] = flat_list
    
    feature_cols = [s for s in perm_imp_results_df.columns if feature in s]
    #Aggregating the mean and SD from the 5 random runs
    perm_imp_results_df[feature + '_' + 'mean'] = perm_imp_results_df[feature_cols].apply(pd.to_numeric, args=['coerce']).mean(axis=1, skipna=False)
    perm_imp_results_df[feature + '_' + 'std'] = perm_imp_results_df[feature_cols].apply(pd.to_numeric, args=['coerce']).std(axis=1, skipna=False)  
            
def subject_gen_perm_imp_main_setup(data, model, stridesList, labelsList, subject_params, config_path):
    '''
    Main setup for the permutation feature importance for subject gen 
    Reference: https://christophm.github.io/interpretable-ml-book/feature-importance.html#fn35
    '''
    lpath, perm_imp_results_df = subject_gen_perm_imp_initial_setup(config_path)
    #12 Feature groups to explore the importance for 
    #features = ["all", "wearables", "kinetic", "spatial_kinetic", "spatial", "spatiotemporal", "temporal_kinetic", "temporal"]
    features = ['stride_length', 'stride_width', 'LeftFPA', 'RightFPA', 'stride_time', 'swing_time', 'stance_time', 'SS_R', 'DS_L',\
        'DS_R', 'cadence', 'stride_speed', 'walk_ratio', 'force_HSR', 'force_MidSSR', 'force_TOR', 'force_HSL', 'force_TOL',\
        'force_MidSSL', 'Butterfly_x_abs', 'ButterflySQ_x']
    global runs 
    runs = 0
    for feature in features:
        #For all 12 feature groups 
        print ('Running for ', feature)
        subject_gen_perm_imp_single_feature(feature, data, model, stridesList, labelsList, subject_params, perm_imp_results_df, lpath)
    display(perm_imp_results_df)
    #Saving all the 7*12 columns for all 12 feature groups and all 5 runs+2(mean/std)
    perm_imp_results_df.to_csv(lpath + 'Permutation_importance_all_results.csv')
    result_mean_cols = [s for s in  perm_imp_results_df.columns if 'mean' in s]
    result_std_cols = [s for s in  perm_imp_results_df.columns if 'std' in s]
    main_result_cols = result_mean_cols + result_std_cols
    #Saving only the mean and std per 12 feature groups (24 columns) that will be used to plot FI later
    perm_imp_results_df[main_result_cols].to_csv(lpath + 'Permutation_importance_only_main_results.csv')
class PermuteTransform():
    '''
    Class for permutation for features of interest for the testing folds with model trained on original features in the training folds 
    '''
    def __init__(self, feature_indices_):
        self.feature_indices_ = feature_indices_
    def transform(self, X, y=None):
        '''
        Shuffle the desire features across the entire testing set 
        '''
        global runs
        np.random.seed(runs)
        shuffled = copy.deepcopy(X)
        for feat_index in self.feature_indices_:
            np.random.shuffle(shuffled[:,:,feat_index])

        runs += 1
        return shuffled
        #return X
    def fit_transform(self, X, y=None):
        '''
        We do not need to shuffle the training set, so we return it as is
        '''
        return X