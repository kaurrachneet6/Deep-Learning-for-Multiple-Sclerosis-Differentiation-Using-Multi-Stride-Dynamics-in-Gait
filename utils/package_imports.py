#Package imports 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from IPython.display import display, HTML
import copy
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve
from inspect import signature
from scipy import interp
#from pyitlib import discrete_random_variable as drv
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, roc_curve
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
#import xgboost 
import joblib
from sklearn.metrics import make_scorer
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.functional import dropout
import random
from torch.utils.data import Dataset, DataLoader
from skorch import NeuralNetClassifier
from skorch import helper
from skorch import dataset
from skorch.callbacks import EarlyStopping
from skorch.callbacks import LRScheduler
from skorch.callbacks import EpochScoring
from skorch.callbacks import Callback
from skorch.callbacks import Checkpoint
from skorch.callbacks import TrainEndCheckpoint
from torch.autograd import Variable
from torch.nn import Parameter
import itertools
from ast import literal_eval

import pickle
import os

#Paths for traditional model results 
results_path_task_generalize_trad = 'results//task_generalize_traditional//'
results_path_subject_generalize_trad = 'results//subject_generalize_traditional//'

#Paths for LSTM results
results_path_task_generalize_lstm = 'results//task_generalize_lstm//'
results_path_subject_generalize_lstm = 'results//subject_generalize_lstm//'
