# predictionNeuralNet
Recurrent Neural Networks for Predicting Household Electric Consumption

Deep learning techniques to predict future household electric consumption. Past one year of household electric consumption data is used for training. Though data only has active power, reactive power, voltage and current values. Input data was only scaled and no other processing was performed.

Second run had power consumption and temperature as well as input features. Mean normalization of both was done for feeding to CNN & LSTM.

Results show good predictions can be obtained from LSTM networks with long time sequences. Combining CNN with LSTM is a good option for some applications. Reactive power does not have Gaussian distribution and thus poor prediction accuracy for reactive power.

it is important to tune training process to account for seasonal power consumption patterns, holidays, weekends vs weekday consumption patterns, etc.

See more details at:
https://www.linkedin.com/pulse/predict-future-electric-consumption-machine-learning-part-2-singh

https://www.linkedin.com/in/sarab96
