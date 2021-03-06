# Comparing-Supervised-Classification-Algorithms
Comparing computational and predictive perfromance of learnining algorithms
 
Importing data and libraries
As a first step data set was imported using pandas library of python, then various required libraries like math, time, StratifiedKFold, svm, LogisticRegression, Kneighbors-Classifier, confusion_matrix were imported.
Discussion
After importing the libraries, data was divided into ten folds using a stratifiedkfold library. So, each fold contains the same number of instances (460). Among these ten folds of data, nine folds were used for training, and one-fold was used for testing purposes. This process is repeated for ten times, where every time the testing fold was changed. Such that for the first time, the first fold was used for testing. For the second time, the second fold was used for testing like this the process continues for ten times. During this process, every time the training of data is performed with three algorithms, namely Support Vector Machine Classification, Logistic regression, K Nearest Neighbour. The performance of these three selected algorithms was compared based upon measures like Accuracy, F-measure(recall, precision), and training time is taken. Comparision was done based on the Friedman test and Nemenyi test(if required). Finally, concluding whether these three algorithms perform similarly or which algorithm performs better if they don't perform similarly.
Algorithm Selection
Support Vector Machine classification, K-nearest neighbor, and logistic regression are quite often used algorithms for most of the machine learning problems. Each algorithm has its own advantages and disadvantages for the particular type of data. Such that SVM is very efficient for high dimensional data, but it is not suitable for the very large data sets. On the other hand, KNN is very simple and intuitive and even works well for data that contain large samples. But, KNN is computationally expensive, and choosing k(number of neighbors to be selected for estimation) is tricky. Logistic regression is very easy and quick to train, gives very fast results. So, in order to compare each algorithm over the given spam or not classification data based upon the factors like accuracy, recall, precision and training time taken for training, and to understand the behavior of these algorithms, I have taken these three algorithms.
Comparisons
Various comparisons taken are computational time, accuracy, recall, precision, and F-measure. From the confusion matrix, tp(true positive), tn(true negative), fp(false positive), fn(false negative) were taken. Then accuracy, recall, precision were calculated by using the following formulas.
accuracy = (tp + tn)/(tp + tn + fp + fn)
recall = tp/(tp + fn)
precision = tp/(tp +  fp)
f-measure  = (2 * recall * precision)/(recall + precision)

These all values are taken for each algorithm,  for each fold. Training time taken by each algorithm for each fold was also considered.
Ranks Assignment
After calculating the values for various measures such as accuracy, recall, precision, f-measure, time taken, ranks are assigned for three algorithms for each fold such that for accuracy, recall, precision, f-measure rank one is given to the algorithm which has the highest value. Rank three is given to the algorithm, which has the lowest value among 3. Rank 2 is given to the algorithm, which is in between rank one and rank three algorithm.
Friedman and Nemenyi test
Friedman test is done for all the measures such as Accuracy, time, recall, precision, and F-measure [1].
From the calculated ranks, mean for all ranks of each algorithm were found, from mean ranks, average ranks, the sum of squared differences(Friedman statistic) were found using the following formulas.
Average rank = 1/nk(∑_ij▒Rij) = (k+1) / 2
Sum of squared differences  =  n∑_j▒〖(Rj-¯(R))〗^2 
After calculating the friedman statistic, it is compared with the critical value (depend upon the k and n values). If the friedman statistic is greater than the critical value (= 7.8 in our case), then the null hypothesis is rejected else failed to reject the null hypothesis [1].
If the null hypothesis is rejected, then the Nemenyi test is performed to find which algorithm is performing significantly different from others. This is done by calculating the Critical difference (CD) from the following formula
CD = q_α √((k(k+1))/6n)
If the difference of the average ranks of two algorithms exceeds the critical difference, then those two algorithms don't work similarly.
Results
All the results were printed in the output logs as exactly as table 12.4,12.8 in [1]
Accuracy
Average rank for SVM :  2.9
Average rank for logistic regression : 1.0
Average rank for KNN : 2.1
Sum of squared difference : 18.2
Null Hypothesis is rejected: All the three algorithms don't perform equally,i.e SVM, logistic regression and logistic regression, KNN won’t work similarly
F-Measure
Average rank for SVM :  3.0
Average rank for logistic regression : 1.0
Average rank for KNN : 2.0
sum of squared differences  20.0
Null Hypothesis is rejected: All the three algorithms don't perform equally,i.e SVM and logistic regression won’t work similarly
Time
Average rank for SVM :  3.0
Average rank for logistic regression : 1.0
Average rank for KNN : 2.0
sum of squared differences  20.0
Null Hypothesis is rejected: All the three algorithms don't perform equally, SVM and KNN won’t work similarly

References
[1]	P. Flach, “Machine Learning,” p. 416.

 

