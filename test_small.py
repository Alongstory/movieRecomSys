from time import time
import sys,argparse
from pyspark import SparkContext
from operator import add
from pyspark.mllib.recommendation import ALS
import math

def test(train_set, test_set, rank, reg_para, partitionNum):
    train_set = train_set.repartition(partitionNum).cache()
    test_predict = test_set.map(lambda x : (x[0], x[1])).partitionBy(partitionNum).cache()
    iteration = 10
    model = ALS.train(train_set, rank, iterations = iteration, lambda_ = reg_para)
    prediction = model.predictAll(test_predict).map(lambda x : ((x[0], x[1]), x[2]))  
    pred_rate = test_set.map(lambda x : ((int(x[0]), int(x[1])), float(x[2])))\
                        .join(prediction, numPartitions = partitionNum) 
    error = math.sqrt(pred_rate.map(lambda x : (x[1][0] - x[1][1]) ** 2).mean())
    return error

if __name__ == '__main__':
    sc = SparkContext(appName = 'Model Test')
    rank = 8
    partitionNum = 10
    reg_para = 0.1
    train_set = sc.textFile('small_train').map(eval)
    test_set = sc.textFile('small_test').map(eval)
    error = test(train_set, test_set, rank, reg_para, partitionNum)
    print '*****************************************************************'
    print 'For testing data the RMSE is %s' % (error)
    print '*****************************************************************'
