sFilt
=====

Spam Filtering using ML

The project focuses on development of an email spam filter which learns to classify among spam and non-spam emails employing various machine learning techniques. A machine learning system would be trained on email messages to learn to distinguish between spam and non-spam messages. Learning here represents the act of observing the spam text patterns building upon a good enough model that makes predictions which fits to the training set well and generalize well on new data.

We employed different supervised classification based algorithmic techniques based on machine learning techniques to train the system on certain data set of pre-categorized spam and non-spam messages and contrast their performance on the test data sets to find an optimal technique with better filtering ability. Our approach dived into different flavors of machine learning algorithms. First we implemented Perceptron algorithm which is based on Hyperplane based Classification model. We then took the initiative to check for performance of instance based learning model which we implemented in K nearest neighbors algorithm. Our final approach was based on Probabilistic model for which we implemented Naïve Bayes algorithm. We generated the feature vectors out of the raw text based dataset.

For each of the learning technique we prepared a training set whose classification labels are also provided to the algorithm and expect it to return good enough predictions on the test set. Then we contrasted the performance of these algorithms by comparing mean error rates, learning
rate and false positive ratio. Analyzing the performance of respective algorithms with the applicability of real life scenario of limit of computing resources, we were able to conclude our results.

We processed the email messages from the dataset and extracted features from it in the form of frequencies of occurrence of words. Then the formed feature vectors along with the labels based on the classification is passed to the classifier. The classifier selects a certain portion of data to train the system. It passes the required training data along with the labels to the system. The system according to the technique of learning employed forms the optimal hypothesis to guess the labels for new data to be encountered in the test set.

During the tests, the algorithms predictions were evaluated and compared with the original labels in the dataset and their performance is analyzed. Our perceptron algorithm employed the Online Learning methodology in which the algorithm is fed the correctness/incorrectness of its predictions and the algorithm is allowed to update its hypothesis accordingly.

Source Code Overview:

The directory tree is as follows:

Project
   |---Data
   
   |---KNN
   
   |---NaiveBayes
   
   |---Perceptron


The "Data" directory contains:
 - The original email messages from Enron dataset:
   Data\enron1\ham
   Data\enron1\spam
  
 - The Python script which we wrote, process.py, which creates pairs of training set & test set of feature vectors:
   Data\enron1\process.py
   
 - The pairs of training set & test set of feature vectors, along with "stats" file which describes which words were chosen to participate in the feature vector. There are 10 pairs per training fraction. For example for training fraction 0.1 and run #0:
   Data\enron1\test_0.1_0.txt
   Data\enron1\train_0.1_0.txt
   Data\enron1\stats_0.1_0.txt
   
Each of the other directories contains the relevant files per algorithm. The relevant files are matlab code (.m), saved data vectors (.mat) and graphs (.fig) generated by this code.
Each file whose name stars with "main" is an executable matlab script.
For example, the executable scripts of KNN algorithm:
main_knn.m
main_knn_test_k.m
main_knn_test_dim.m
main_knn_test_majority_factor.m
