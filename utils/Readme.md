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
