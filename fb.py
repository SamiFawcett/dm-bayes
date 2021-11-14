import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal


def accuracy(estimated_classes, actual_classes):
    ccc = [0, 0, 0, 0]  #class correct counts
    cic = [0, 0, 0, 0]  #class incorrect counts
    for i in range(len(estimated_classes)):
        ec = estimated_classes[i]
        ac = actual_classes[i]
        if (ec == ac):
            ccc[ac] += 1
        else:
            cic[ac] += 1

    ca = []
    recall = 0
    for i in range(4):
        ca.append(ccc[i] / (ccc[i] + cic[i]))
        recall += ca[i] * (ccc[i] + cic[i])
    recall = recall / sum(np.array(ccc) + np.array(cic))

    return [ca, recall]


def splitIntoClassDatasets(point_view, response_column):
    classes = [0, 1, 2, 3]
    datasets = []
    c0 = []
    c1 = []
    c2 = []
    c3 = []

    for i in range(0, len(response_column)):
        if (response_column[i] == 0):
            c0.append(point_view[i])
        elif (response_column[i] == 1):
            c1.append(point_view[i])
        elif (response_column[i] == 2):
            c2.append(point_view[i])
        elif (response_column[i] == 3):
            c3.append(point_view[i])
    datasets.append(c0)
    datasets.append(c1)
    datasets.append(c2)
    datasets.append(c3)
    return datasets


def pRange(point_view, lower_bound, upper_bound):
    p_subset = []
    for i in range(lower_bound, upper_bound + 1):
        p_subset.append(point_view[i])

    return np.array(p_subset)


def multiClassClassification(column_view):
    response_column = column_view[0]
    new_response_column = []
    for val in response_column:
        if (val <= 40):
            new_response_column.append(0)
        elif (val <= 60):
            new_response_column.append(1)
        elif (val <= 100):
            new_response_column.append(2)
        else:
            new_response_column.append(3)

    return np.array(new_response_column)


def computePriors(classDatasets, datasetSize):
    priors = []
    for c in classDatasets:
        classSize = len(c)
        priors.append(classSize / datasetSize)
    return priors


def computeLikelihood(x, c, cov_c):
    c_col_view = np.transpose(c)
    mean_vector = []
    for col in c_col_view:
        mean_vector.append(np.mean(col))

    return multivariate_normal.pdf(x, mean=mean_vector, cov=cov_c)


def bayes_classifer(training_data, training_response, testing_data,
                    testing_response):
    class_datasets = splitIntoClassDatasets(training_data, training_response)
    class_priors = computePriors(class_datasets, len(training_data))
    class_cardinalities = []
    class_means = []
    class_covariances = []
    for c in class_datasets:
        class_cardinalities.append(len(c))
        c_mean = []
        col_view_c = np.transpose(c)
        for col in col_view_c:
            c_mean.append(np.mean(col))
        class_means.append(np.array(c_mean))
        class_covariances.append(np.cov(np.transpose(c)))

    estimated_classes = []
    for point in testing_data:
        #class probability estimations
        probabilities = []
        for i in range(len(class_datasets)):
            c = class_datasets[i]
            m = class_means[i]
            co = class_covariances[i]
            p_hat = multivariate_normal.pdf(point, mean=m, cov=co)
            probabilities.append(p_hat)
        #max selection
        idx = 0
        highest_prob = 0
        estimated_class = 0
        for p in probabilities:
            if (p > highest_prob):
                estimated_class = idx
                highest_prob = p
            idx += 1
        estimated_classes.append(estimated_class)
    return estimated_classes


def parse(filename):
    headers = list(pd.read_csv(filename, nrows=0).columns)
    headers.pop(len(headers) - 1)
    headers.pop(0)
    data = pd.read_csv(filename, usecols=headers).to_numpy()
    unscaled_column_view = list(np.transpose(data))
    response_column = multiClassClassification(unscaled_column_view)
    unscaled_column_view.pop(0)
    unscaled_column_view = np.array(unscaled_column_view)
    point_view = np.transpose(unscaled_column_view)

    return [unscaled_column_view, response_column, point_view, headers]


if __name__ == "__main__":

    filename = 'energydata_complete.csv'
    column_view, response_column, point_view, headers = parse(filename)

    training_data = pRange(point_view, 0, 14734)
    training_response = pRange(response_column, 0, 14734)
    testing_data = pRange(point_view, 14735, 14735 + 4999)
    testing_response = pRange(response_column, 14735, 14735 + 4999)

    estimated_classes = bayes_classifer(training_data, training_response,
                                        testing_data, testing_response)

    class_accruracy, recall = accuracy(estimated_classes, testing_response)

    print("class accuracies [c0, c1, c2, c3]: ", class_accruracy)
    print("recall: ", recall)
