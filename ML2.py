#importing required libraries
import pandas as pd
import math
import time
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

dataset = pd.read_csv('spambase.data')

dataset_list = dataset.values.tolist()
dataset_input_list = dataset.iloc[:,:-1].values.tolist()
dataset_output_list = dataset.iloc[:,-1].values.tolist()


skf = StratifiedKFold(n_splits=10)
skf.get_n_splits(dataset_input_list,dataset_output_list)
StratifiedKFold(n_splits=10, random_state=None, shuffle=False)

algo_1 = []
algo_2 = []
algo_3 = []

algo_1_time = []
algo_2_time = []
algo_3_time = []

algo_1_recall = []
algo_2_recall = []
algo_3_recall = []

algo_1_precision = []
algo_2_precision = []
algo_3_precision = []

algo_1_fmeasure = []
algo_2_fmeasure = []
algo_3_fmeasure = []

algo_1_ranks = []
algo_2_ranks = []
algo_3_ranks = []

for train_index, test_index in skf.split(dataset_input_list,dataset_output_list):
    
    training_fold_list = []
    testing_fold_list = []
    training_fold_input_list = []
    training_fold_output_list = []
    testing_fold_input_list = []
    testing_fold_output_list = []
    
    for i in train_index:
        training_fold_list.append(dataset_list[i])
        training_fold_input_list.append(dataset_list[i][:-1])
        training_fold_output_list.append(dataset_list[i][-1])
        
    for j in test_index:
        testing_fold_list.append(dataset_list[j])
        testing_fold_input_list.append(dataset_list[j][:-1])
        testing_fold_output_list.append(dataset_list[j][-1])
        
    
    
      
    #training with SVM algorithm
    svmclf = svm.SVC(gamma = 'scale')
    time_start = time.clock() 
    svmclf.fit(training_fold_input_list,training_fold_output_list)
    time_elapsed = time.clock() - time_start
    svmresult_list = svmclf.predict(testing_fold_input_list)
    algo_1_time.append(time_elapsed)

    
    time_start = time.clock()
    #training with logistic regression
    lgsregression = LogisticRegression(random_state=0,solver='lbfgs').fit(training_fold_input_list,training_fold_output_list)
    time_elapsed = time.clock() - time_start
    lgsregression_result = lgsregression.predict(testing_fold_input_list) 
    algo_2_time.append(time_elapsed)
    
    
    #training with K Nearest Neighbour
    knn = KNeighborsClassifier(n_neighbors = 5)
    time_start = time.clock()
    knn.fit(training_fold_input_list,training_fold_output_list)
    time_elapsed = time.clock() - time_start
    knn_result = knn.predict(testing_fold_input_list)
    
    algo_3_time.append(time_elapsed)
    
    #calculating recall, precision and fmeasure for SVM algorithm
    tn_svm, fp_svm, fn_svm, tp_svm = confusion_matrix(testing_fold_output_list,svmresult_list).ravel()
    algo_1.append((tp_svm+tn_svm)/(tn_svm + tp_svm + fn_svm + fp_svm))
    algo_1_recall.append(tp_svm/(tp_svm + fn_svm))
    algo_1_precision.append(tp_svm/(tp_svm + fp_svm))
    recall_1 = tp_svm/(tp_svm + fn_svm)
    precision_1 = tp_svm/(tp_svm + fp_svm)
    
    #calculating recall, precision and fmeasure for logistic regression algorithm
    tn_lgs, fp_lgs, fn_lgs, tp_lgs = confusion_matrix(testing_fold_output_list,lgsregression_result).ravel()
    algo_2.append((tp_lgs + tn_lgs)/(tn_lgs + tp_lgs + fn_lgs + fp_lgs))
    algo_2_recall.append(tp_lgs/(tp_lgs + fn_lgs))
    algo_2_precision.append(tp_lgs/(tp_lgs + fp_lgs))
    recall_2 = tp_lgs/(tp_lgs + fn_lgs)
    precision_2 = tp_lgs/(tp_lgs + fp_lgs)
    
    #calculating recall, precision and fmeasure for K Neareest Neighbour algorithm
    tn_knn, fp_knn, fn_knn, tp_knn = confusion_matrix(testing_fold_output_list,knn_result).ravel()
    algo_3.append((tp_knn + tn_knn)/(tn_knn + tp_knn + fn_knn + fp_knn))
    algo_3_recall.append(tp_knn/(tp_knn + fn_knn))
    algo_3_precision.append(tp_knn/(tp_knn + fp_knn))
    recall_3 = tp_knn/(tp_knn + fn_knn)
    precision_3 = tp_knn/(tp_knn + fp_knn)
    
    algo_1_fmeasure.append((2*recall_1*precision_1)/(recall_1 + precision_1))
    algo_2_fmeasure.append((2*recall_2*precision_2)/(recall_2 + precision_2))
    algo_3_fmeasure.append((2*recall_3*precision_3)/(recall_3 + precision_3))
    
def friedmanNemeyi():
    avg_R = 0
    sum_R = 0
    for i  in range(10):
        sum_R = sum_R + algo_1_ranks[i] + algo_2_ranks[i]+ algo_3_ranks[i]
    avg_R = sum_R/(10*3)   
    sum_of_squared_differences = 10 * ( ((sum(algo_1_ranks)/10 - avg_R) * (sum(algo_1_ranks)/10 - avg_R)) + ((sum(algo_2_ranks)/10 - avg_R) * (sum(algo_2_ranks)/10 - avg_R)) + ((sum(algo_3_ranks)/10 - avg_R) * (sum(algo_3_ranks)/10 - avg_R)) )

    print(" Rank mean ",avg_R)
    print(" sum of squared differences ",sum_of_squared_differences)
    print()
    if(sum_of_squared_differences > 7.8): #7.8 is the critical value for three algorithms with 10 folds of data
        print(" Null Hypothesis is rejected: All the three algorithms don't perform equally")
        k = 3
        CD = 2.343 * math.sqrt((k * ( k + 1 ))/(6 * 10)) #k = 3 and n = 10
        print(' Critical Difference ',CD)
        print()
        if abs(sum(algo_1_ranks)/10 - sum(algo_2_ranks)/10) > CD:
            print(' difference between SVM and logistic regression exceeds critical difference')
        if abs(sum(algo_2_ranks)/10 - sum(algo_3_ranks)/10) > CD:
            print(' difference between logistic regression and KNN exceeds critical difference')
        if abs(sum(algo_1_ranks)/10 - sum(algo_3_ranks)/10) > CD:
            print(' difference between SVM and KNN exceeds critical difference')
        print()
    else:
        print(" Failed to reject null Hypothesis : All the algorithms performs equally")
    

def displayFunc(x_1,x_2,x_3):
    print("Fold  ","      SVM     ","   logistic regression    ","   KNN   ")
    for i in range(10):
       if i == 9:
           print (' {:1d}        {:0.4f}           {:0.4f}                {:0.4f}'.format(i+1, x_1[i],x_2[i],x_3[i]))
           continue
       print (' {:1d}         {:0.4f}           {:0.4f}                {:0.4f}'.format(i+1, x_1[i],x_2[i],x_3[i]))
    print()
    print (' avg       {:0.4f}           {:0.4f}                {:0.4f}'.format(sum(x_1)/10, sum(x_2)/10, sum(x_3)/10 ))
    print()
    print()
    print("Friedman test and results")
    print("Fold  ","      SVM     ","   logistic regression    ","        KNN   ")
    for i in range(10):
        if i == 9:
            print (' {:1d}        {:0.4f}({:1d})           {:0.4f}({:1d})              {:0.4f}({:1d})'.format(i+1, x_1[i],algo_1_ranks[i],x_2[i],algo_2_ranks[i], x_3[i],algo_3_ranks[i]))
            continue
        print (' {:1d}         {:0.4f}({:1d})           {:0.4f}({:1d})              {:0.4f}({:1d})'.format(i+1, x_1[i],algo_1_ranks[i],x_2[i],algo_2_ranks[i],x_3[i],algo_3_ranks[i]))
    print()
    print (' avg rank    {:0.1f}                 {:0.1f}                    {:0.1f}'.format(sum(algo_1_ranks)/10, sum(algo_2_ranks)/10, sum(algo_3_ranks)/10 ))
    print()
    friedmanNemeyi()
 
#Assigning ranks for Accuracy, recall, precision, f-measure and Time
def rankAssign(x_1,x_2,x_3,measure):
    print(measure)
    if measure == 'Time':
        for i in range(10):
            if x_1[i] <= x_2[i]:
                if x_1[i] <= x_3[i]:
                    algo_1_ranks.append(1)
                    if x_2[i] <= x_3[i]:
                        algo_2_ranks.append(2)
                        algo_3_ranks.append(3)
                    else:
                        algo_2_ranks.append(2)
                        algo_3_ranks.append(3)
            if x_2[i] <= x_3[i]:
                if x_2[i] <= x_1[i]:
                    algo_2_ranks.append(1)
                    if x_1[i] <= x_3[i]:
                        algo_1_ranks.append(2)
                        algo_3_ranks.append(3)
                    else:
                        algo_3_ranks.append(2)
                        algo_1_ranks.append(3)
            if x_3[i] <= x_1[i]:
                if x_3[i] <= x_2[i]:
                    algo_3_ranks.append(1)
                    if x_1[i] <= x_2[i]:
                        algo_1_ranks.append(2)
                        algo_2_ranks.append(3)
                    else:
                        algo_2_ranks.append(2)
                        algo_1_ranks.append(3)
    else:
        for i in range(10):
            if x_1[i] >= x_2[i]:
                if x_1[i] >= x_3[i]:
                    algo_1_ranks.append(1)
                    if x_2[i] >= x_3[i]:
                        algo_2_ranks.append(2)
                        algo_3_ranks.append(3)
                    else:
                        algo_2_ranks.append(2)
                        algo_3_ranks.append(3)
            if x_2[i] >= x_3[i]:
                if x_2[i] >= x_1[i]:
                    algo_2_ranks.append(1)
                    if x_1[i] >= x_3[i]:
                        algo_1_ranks.append(2)
                        algo_3_ranks.append(3)
                    else:
                        algo_3_ranks.append(2)
                        algo_1_ranks.append(3)
            if x_3[i] >= x_1[i]:
                if x_3[i] >= x_2[i]:
                    algo_3_ranks.append(1)
                    if x_1[i] >= x_2[i]:
                        algo_1_ranks.append(2)
                        algo_2_ranks.append(3)
                    else:
                        algo_2_ranks.append(2)
                        algo_1_ranks.append(3)
    displayFunc(x_1,x_2,x_3)



rankAssign(algo_1,algo_2,algo_3,'Accuracy')

algo_1_ranks = []
algo_2_ranks = []
algo_3_ranks = []

rankAssign(algo_1_recall,algo_2_recall,algo_3_recall,'Recall')
algo_1_ranks = []
algo_2_ranks = []
algo_3_ranks = []

rankAssign(algo_1_precision,algo_2_precision,algo_3_precision,'Precision')
algo_1_ranks = []
algo_2_ranks = []
algo_3_ranks = []

rankAssign(algo_1_fmeasure,algo_2_fmeasure,algo_3_fmeasure,'F-measure')
algo_1_ranks = []
algo_2_ranks = []
algo_3_ranks = []

rankAssign(algo_1_time,algo_2_time,algo_3_time,'Time')
algo_1_ranks = []
algo_2_ranks = []
algo_3_ranks = []
    


        

