import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

# load mini training data and labels
mini_train = np.load('knn_minitrain.npy')
mini_train_label = np.load('knn_minitrain_label.npy')

# randomly generate test data
mini_test = np.random.randint(20, size=20)
mini_test = mini_test.reshape(10,2)


# Define knn classifier
def kNNClassify(newInput, dataSet, labels, k):

    result=[]

    ########################
    # Input your code here #
    ########################

    # Range is (0,10) because of the testing size
    for x in range(0, 10):

        # Array to hold the distances
        distance=np.array([])

        # Calculate the distance between two given points =  sqrt((x-y)^2 + .....)
        for y in range(0, 40):
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
        distance = distance.reshape(40, 2)

        # Sort the distances in the array
        distance = distance[distance[:, 0].argsort(kind='quicksort')]

        # Assign labels for classification
        neighbors = [0] * k

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
        for i in range(0, k):
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

outputlabels=kNNClassify(mini_test,mini_train,mini_train_label,4)

print ('random test points are:', mini_test)
print ('knn classified labels for test:', outputlabels)

# plot train data and classified test data
train_x = mini_train[:,0]
train_y = mini_train[:,1]
fig = plt.figure()
plt.scatter(train_x[np.where(mini_train_label==0)], train_y[np.where(mini_train_label==0)], color='red')
plt.scatter(train_x[np.where(mini_train_label==1)], train_y[np.where(mini_train_label==1)], color='blue')
plt.scatter(train_x[np.where(mini_train_label==2)], train_y[np.where(mini_train_label==2)], color='yellow')
plt.scatter(train_x[np.where(mini_train_label==3)], train_y[np.where(mini_train_label==3)], color='black')

test_x = mini_test[:,0]
test_y = mini_test[:,1]
outputlabels = np.array(outputlabels)
plt.scatter(test_x[np.where(outputlabels==0)], test_y[np.where(outputlabels==0)], marker='^', color='red')
plt.scatter(test_x[np.where(outputlabels==1)], test_y[np.where(outputlabels==1)], marker='^', color='blue')
plt.scatter(test_x[np.where(outputlabels==2)], test_y[np.where(outputlabels==2)], marker='^', color='yellow')
plt.scatter(test_x[np.where(outputlabels==3)], test_y[np.where(outputlabels==3)], marker='^', color='black')

# save diagram as png file
plt.savefig("miniknn.png")
