# Training-Neural-Networks-via-the-Alternating-Direction-Method-of-Multipliers-on-Apache-Spark

# Abstract
Many problems of recent interest in statistics and machine learning can be posed in the framework of convex optimization. Due to the explosion in size and complexity of modern datasets, ‘Big Data’ becomes increasingly necessary for data handling and neural networks become important to train large network models. This project experimented with an unconventional training method that uses alternating direction methods and Bregman iteration to train networks without gradient descent steps, on top of Apache Spark. The results are compared by the loss and accuracy plots of train, validation, and test model performances.
**Apache Spark installation version 3.0**

# Experimental Results
The experiment was done on MNIST data. The MNIST dataset is split into three parts, 55,000 data points of training data (mnist.train), 10,000 points of test data (mnist.test), and 5,000 points of validation data (mnist.validation). Every MNIST data point has two parts: an image of a handwritten digit and a corresponding label.
The results are illustrated based on loss and accuracy of model training, validation, and testing performances accordingly.
![alt An ilustration of training(blue), validation(green) and testing(red) Accuracies w.r.to number of epochs traversal.](https://github.com/dastagiri7/Training-Neural-Networks-via-the-Alternating-Direction-Method-of-Multipliers-on-Apache-Spark/blob/main/ACC_after_20_epoch.png)
Figure 1. An ilustration of training(blue), validation(green) and testing(red) Accuracies w.r.to number of epochs traversal.

The **85 percent** of testing accuracy reached at the 10th epoch.

![alt An ilustration of training(blue), validation(green) and testing(red) Losses w.r.to number of epochs traversal.](https://github.com/dastagiri7/Training-Neural-Networks-via-the-Alternating-Direction-Method-of-Multipliers-on-Apache-Spark/blob/main/LOSS_after_20_epoch.png)
Figire 2. An ilustration of training(blue), validation(green) and testing(red) Losses w.r.to number of epochs traversal.

## Brief Note
Initially, I thought of developing the ADMM implementation on top of Hadoop distributed file system(HDFS) and MapRed. Since Hadoop-ecosystem allows large data replications in multiple machines and MapRed allows training neural network modeling in a distributed way. But, because of the iterative functionality of ADMM, I thought it was not possible to aggregate the results using Reducer’s (not experimented). 
Apache Spark, which is another big data open-source framework, allows to play with huge datasets and parallelized as block partitions in-memory. In this empirical project, most of the time was allocated to research work related to ADMM(equations) and Spark optimization. The final results came pretty satisfactorily.
In future works, continue this work by upgrading the model with graph neural networks and scale the ADMM optimization. And, experiment once, the distributed ADMM mechanism on Hadoop-ecosystem.
