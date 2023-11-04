#-------------------------------------------------------------------------
# AUTHOR: Drake Fafard
# FILENAME: svm.py
# SPECIFICATION: Find the optimal SVM classifier for a given data set.
# FOR: CS 4210- Assignment #3
# TIME SPENT: 1 hour on question 4.
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd

#defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the training data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] #getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the training data by using Pandas library

X_test = np.array(df.values)[:,:64] #getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] #getting the last field to create the class testing data and convert them to NumPy array

#created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
#--> add your Python code here

highest_accuracy = -1
c_val_highest = 1
degree_val_highest = 1
kernel_val_highest = "linear"
decision_function_shape_val_highest = "ovo"

for c_val in c:
    for degree_val in degree:
        for kernel_val in kernel:
            for decision_function_shape_val in decision_function_shape:

                #Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape.
                #For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                #--> add your Python code here

                svm_model = svm.SVC(C=c_val, degree=degree_val, kernel=kernel_val, decision_function_shape=decision_function_shape_val)

                #Fit SVM to the training data
                #--> add your Python code here

                svm_model.fit(X_training, y_training)

                #make the SVM prediction for each test sample and start computing its accuracy
                #hint: to iterate over two collections simultaneously, use zip()
                #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
                #to make a prediction do: clf.predict([x_testSample])
                #--> add your Python code here

                total_correct_predictions = 0
                total_predictions = 0
                accuracy = 0

                for (x_testSample, y_testSample) in zip(X_test, y_test):
                    y_pred = svm_model.predict([x_testSample])
                    if y_pred == y_testSample:
                        total_correct_predictions += 1
                    total_predictions += 1

                if total_predictions > 0:
                    accuracy = total_correct_predictions / total_predictions

                #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                #with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                #--> add your Python code here

                if accuracy > highest_accuracy:
                    highest_accuracy = accuracy
                    c_val_highest = c_val
                    degree_val_highest = degree_val
                    kernel_val_highest = kernel_val
                    decision_function_shape_val_highest = decision_function_shape_val

                # print("Total predictions: " + str(total_predictions))
                # print("Total correct predictions: " + str(total_correct_predictions))

print("Highest SVM accuracy so far: " + "{:.2f}".format(highest_accuracy) + ", Parameters: C = " + str(c_val_highest), end="")
print(", degree = " + str(degree_val_highest) + ", kernel = " + str(kernel_val_highest), end="")
print(", decision_function_shape = " + str(decision_function_shape_val_highest))

