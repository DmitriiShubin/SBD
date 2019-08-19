## Sub-Band Decomposition for the time-series Deep Learning models



NOTE: Full description of the project will be added later.


Python/ Matlab project for detection abnormal ECG signals.

Final performance: AUC-ROC 0.985

Dataset: https://www.kaggle.com/shayanfazeli/heartbeat/home
Download weights of the model (66Mb): https://drive.google.com/file/d/1nNueZK04G0hY1OMzG7zJdWeoAR7xjUnh/view?usp=sharing

Hardware: Laptop Acer Nitro 5, CPU: i7-8750H, GPU: 1050Ti, 16GB RAM +SSD

Required frameworks: Keras, Tensorflow, Sklearn, Pandas, Numpy

Used techniques for accuracy improvement:

1. Stacking of CNNs with different kernel sizes, averaging and learning in one graph 
2. Downsampling up to 85Hz
3. Dropout
4. Snapshot ensembles
5. Ensembling in folds

Total number of models: 50, training time ~ 3 hours
