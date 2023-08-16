#Justin Turbeville
#COM307 - Final Project

import csv
import random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np

def importData(fileName):
    """This function imports the csv file as a 2D array, it manipulates the data
    by removing the mutual funds that are bond funds."""
    dataset = []
    with open(fileName, 'r') as file:
        reader =  csv.reader(file)
        count = 0
        for row in reader:
            if(count == 0):
                row.pop(5)
                features = row
            else:
                if(float(row[5]) < 25):
                    row.pop(5)
                    dataset.append(row)   
            count += 1

    count = 0
    minMax = []
    for data in dataset:
        for i in range(len(data)):
            if(i == 4):
                if(data[i][0] == '#'):
                    data[i] = 0
                else:
                    MDY = data[i].split('/')
                    data[i] = 2021 - int(MDY[2])
            else:
                try:
                    data[i] = float(data[i])
                except(ValueError):
                    pass
            if(count == 0):
                minMax.append([data[i], data[i]])
            else:
                if(data[i] < minMax[i][0]):
                    minMax[i][0] = data[i]
                elif(data[i] > minMax[i][1]):
                    minMax[i][1] = data[i]
        count += 1

    return dataset, features, minMax

def standardize(dataset, minMax, alphaInd):
    """Standardizes values for each feature by turning the values into precentages.
    k-nearest neighbor cannot use the current values."""
    for i in range(len(dataset[0])):
        if(i != alphaInd):
            if(i == 0):
                for data in dataset:
                    if(data[i] == 'Small'):
                        data[i] = 0
                    elif(data[i] == 'Medium'):
                        data[i] = 50
                    else:
                        data[i] = 100
            else:
                diff = minMax[i][1] - minMax[i][0]
                for data in dataset:
                    data[i] = round((data[i] - minMax[i][0]) / diff * 100, 2)
        
    return dataset

def kNN():
    """Creates 'm' random models to be tested using k-nearest neighbor. Displays results."""
    overallBest = []
    m = 1
    for i in range(m):
        fileName = 'Mutual_funds.csv'
        dataset, features, minMax = importData(fileName)
        alphaInd = features.index('alpha')
        dataset = standardize(dataset, minMax, alphaInd)
        featureInds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20]
        f = getRandomFeatures(featureInds, 5, [])
        accuracy, crossMatrix = testNearestNeighbor(f, dataset, 1, 99, alphaInd)

        if(i == 0):
            overallBest = [accuracy, crossMatrix, f]
        else:
            if(accuracy > overallBest[0]):
                overallBest = [accuracy, crossMatrix, f]

    print('Accuracy of best model = ' + str(round(overallBest[0] * 100, 1)) + '%')
    print('Correct positives = ' + str(overallBest[1][0]))
    print('Correct negatives = ' + str(overallBest[1][1]))
    print('False positives = ' + str(overallBest[1][3]))
    print('False negatives = ' + str(overallBest[1][2]))
    print('Features = ' + features[f[0]] + ', ' + features[f[1]] + ', ' +
    features[f[2]] + ', ' + features[f[3]] + ', ' + features[f[4]])

def kNearestNeighbor(point, dataset, features, k, responseInd):
    """Uses k-nearest neighbor and 5 features to predict whether a mutual
    funbd beat the market."""
    distances = []
    for i in range(len(dataset)):
        data = dataset[i]
        totalDist = 0
        for f in features:
            totalDist += abs(point[f] - data[f])**2
        distances.append([i, totalDist])
    distances.sort(key=sortFunction)

    beatMarketCount = 0
    failedCount = 0
    for i in range(k):
        if(dataset[distances[i][0]][responseInd] > 0):
            beatMarketCount += 1
        else:
            failedCount += 1

    if(beatMarketCount > failedCount):
        return True
    else:
        return False

def testNearestNeighbor(featureInds, dataset, testSize, trainingSize, responseInd):
    """Performs cross validation 't' times for a given set of features using k-nearest neighbor.
    Returns the accuracy of the best model."""
    numTestPoints = (len(dataset) * testSize) // (testSize + trainingSize)
    accuracySum = 0
    t = 1
    for i in range(t):
        fileName = 'Mutual_funds.csv'
        dataset, features, minMax = importData(fileName)
        alphaInd = features.index('alpha')
        dataset = standardize(dataset, minMax, alphaInd)
        featureInds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20]
        testData = []
        for j in range(numTestPoints):
            testData.append(dataset.pop(random.randint(0, len(dataset) - 1)))

        correct = 0
        crossMatrix = [0, 0, 0, 0]
        count = 0
        for point in testData:
            count += 1
            prediction = kNearestNeighbor(point, dataset, featureInds, 5, responseInd)
            if(point[responseInd] > 0 and prediction == True):
                correct += 1
                crossMatrix[0] += 1
            elif(point[responseInd] <= 0 and prediction == False):
                correct += 1
                crossMatrix[1] += 1
            else:
                if(prediction == True):
                    crossMatrix[2] += 1
                else:
                    crossMatrix[3] += 1
        accuracySum += correct / numTestPoints
    return accuracySum / t, crossMatrix
    
def sortFunction(pair):
    return pair[1]

def getRandomFeatures(featureInds, num, curFeatures):
    if(num == 0):
        return curFeatures
    else:
        curFeatures.append(featureInds.pop(random.randint(0, len(featureInds) - 1)))
        return getRandomFeatures(featureInds, num - 1, curFeatures)

def regression():
    """Creates 'm' random logistic regression models. Displays results of the most accurate
    regression model as a cross matrix."""
    overallBest = []
    m = 1
    for q in range(m):
        best = []
        featureInds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20]
        f = []
        for j in range(5):
            r = random.randint(0, len(featureInds) - 1)
            f.append(featureInds.pop(r))

        trials = 1
        for i in range(trials):
            """Performs cross validation 't' times and stores the most accurate result."""
            fileName = 'Mutual_funds.csv'
            dataset, features, minMax = importData(fileName)
            alphaInd = features.index('alpha')
            for data in dataset:
                if(data[alphaInd] > 0):
                    data[alphaInd] = 1
                else:
                    data[alphaInd] = 0
                if(data[0] == 'Small'):
                    data[0] = 0
                elif(data[0] == 'Medium'):
                    data[0] = 50
                else:
                    data[0] = 100
            accuracy, crossMatrix = regressionModel(dataset, features, f, alphaInd, 1, 99)
            if(i == 0):
                best = [accuracy, crossMatrix]
            else:
                if(accuracy > best[0]):
                    best = [accuracy, crossMatrix]
        if(q == 0):
            overallBest = [best[0], best[1], f]
        else:
            if(best[0] > overallBest[0]):
                overallBest = [best[0], best[1], f]

    print('Accuracy = ' + str(round(overallBest[0] * 100, 1)))
    print('Features = ' + features[overallBest[2][0]] + ', ' + features[overallBest[2][1]] + ', ' +
    features[overallBest[2][2]] + ', ' + features[overallBest[2][3]] + ', ' + features[overallBest[2][4]])
    sn.heatmap(overallBest[1], annot=True)
    plt.show()

def regressionModel(tempDataset, features, featureInds, alphaInd, testSize, trainingSize):
    """Actually creates a regression model and performs cross validation. Used tutorial from
    GeeksforGeeks to complete this portion."""
    testNum = (len(tempDataset) * testSize) // (testSize + trainingSize)
    xTest = []
    yTest = []
    yTrain = []

    for j in range(testNum):
        r = random.randint(0, len(tempDataset) - 1)
        d = tempDataset.pop(r)
        yTest.append(int(d.pop(alphaInd)))
        xTest.append(d)

    for data in tempDataset:
        yTrain.append(int(data.pop(alphaInd)))

    logisticRegression= LogisticRegression()
    logisticRegression.fit(tempDataset, yTrain)
    yPred=logisticRegression.predict(xTest)

    
    # sn.heatmap(confusion_matrix, annot=True)
    accuracy = metrics.accuracy_score(yTest, yPred)
    return accuracy, pd.crosstab(np.array(yTest), np.array(yPred), rownames=['Actual'], colnames=['Predicted'])

def checkExpense():
    fileName = 'Mutual_funds.csv'
    dataset, features, minMax = importData(fileName)
    alphaInd = features.index('alpha')
    beatTotal = 0
    beatCount = 0
    lostTotal = 0
    lostCount = 0
    for i in range(1, len(dataset)):
        if(dataset[i][alphaInd] > 0):
            beatTotal += dataset[i][2]
            beatCount += 1
        else:
            lostTotal += dataset[i][2]
            lostCount += 1
    print(beatTotal/beatCount)
    print(lostTotal/lostCount)

def main():
    # print("This program will try to predict whether mutual funds will beat the market\n" + 
    # "or not using k-nearest neighbor and logistic regression To model using KNN, press enter.")
    # input('')
    # kNN()
    # print("\nTo see the logistic regression model, press enter.")
    # input("")
    regression()
    # checkExpense()

main()
