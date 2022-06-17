import math
import numpy as np  
from download_mnist import load
import operator  
import time
# classify using kNN  
#x_train = np.load('../x_train.npy')
#y_train = np.load('../y_train.npy')
#x_test = np.load('../x_test.npy')
#y_test = np.load('../y_test.npy')
x_train, y_train, x_test, y_test = load()
x_train = x_train.reshape(60000,28,28)
x_test  = x_test.reshape(10000,28,28)
x_train = x_train.astype(float)
x_test = x_test.astype(float)
def kNNClassify(newInput, dataSet, labels, k):
    result=[]
    ########################
    # Input your code here #
    ########################

    for x in range(0, len(newInput)):

        # Hold the distances
        distance = np.array([])

        # Calculate the distance between two given points =  sqrt((x-y)^2 + .....)
        for y in range(0, 60000):
            # Subtract Testing Point - Training Point
            sub = np.subtract(newInput[x],dataSet[y])
            # Square the result (Testing Point - Training Point) ^ 2
            square= np.power(sub,2)
            # Sum x^2 + y ^ 2
            sum = np.sum(square)
            # Take the resulting value distance and append it to the array
            distance = np.append(distance,np.array([sum*sum]), axis=0)
            # Store the corresponding label to the
            distance = np.append(distance,np.array([labels[y]]),axis=0)

        # Reshape array into a 2d array
        distance = distance.reshape(60000, 2)

        # Sort the distances in the array
        distance = distance[distance[:, 0].argsort(kind='quicksort')]

        # Labels are the digits 0 through 9
        neighbors = [0] * 10

        # Convert the labels into integers
        label = distance[:, 1]
        label = label.astype('int64')

        # Count the labels of the k smallest/nearest distance
        for i in range(0, k):
            neighbors[label[i]] += 1
            i += 1

        # Max
        maximum = neighbors[0]
        index = 0

        # Find the largest/furthest distance
        for i in range(0, 10):
            # Compare distances
            if neighbors[i] > maximum:
                index = i
                # If new max is found replace existing value
                maximum = neighbors[i]
            i += 1

        # Store the result in the result array
        result.append(index)

        # Increase x by 1
        x += 1

    ####################
    # End of your code #
    ####################
    return result

start_time = time.time()
outputlabels=kNNClassify(x_test[0:20],x_train,y_train,20)
result = y_test[0:20] - outputlabels
result = (1 - np.count_nonzero(result)/len(outputlabels))
print ("---classification accuracy for knn on mnist: %s ---" %result)
print ("---execution time: %s seconds ---" % (time.time() - start_time))
