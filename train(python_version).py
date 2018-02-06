# -*- coding: utf-8 -*-
import numpy as np
from time import time
import sys,argparse
from pyspark import SparkContext
from operator import add
from pyspark.mllib.recommendation import ALS
import math

def proc_data(path):
    rating_raw = sc.textFile(path)
    header = rating_raw.take(1)[0]
    rating_data = rating_raw.filter(lambda line : line != header).map(lambda s : s.split(","))\
                            .map(lambda x : (x[0], x[1], x[2])).cache()
    return rating_data

def train(train_set, validation_set, partitionNum):
    # train_set, validation_set -> (user, movie, rating)
    train_set = train_set.repartition(partitionNum).cache()
    iteration = 10
    regularization_paras = [0.01, 0.1, 1.0]
    ranks = [4, 8 ,12]
    errors = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    err_index = 0
    # (user, movie, rating) -> (user, movie)
    validation_predict = validation_set.map(lambda x : (x[0], x[1])).partitionBy(partitionNum).cache()
    min_error = float('inf')
    best_rank = -1
    best_para = 0.0

    # Train model with different parameters
    for para in regularization_paras:
        for rank in ranks:
            start = time()
            model = ALS.train(train_set, rank = rank, iterations = iteration, lambda_ = para)
            prediction = model.predictAll(validation_predict).map(lambda x : ((x[0], x[1]), x[2]))
            # ((user, movie), (rating, predict_rating))
            pred_rate = validation_set.map(lambda x : ((int(x[0]), int(x[1])), float(x[2])))\
                                      .join(prediction, numPartitions = partitionNum)
            error = math.sqrt(pred_rate.map(lambda x : (x[1][0] - x[1][1]) ** 2).mean())
            errors[err_index] = error
            err_index += 1
            now = time() - start
            print 'For rank %d and regularization parameter %d the RMSE is %f, took %s seconds' % (rank, para, error, now)
            # Find the best parameters    
            if error < min_error:
                min_error = error
                best_rank = rank
                best_para = para
    print 'The best model was trained with rank %s and regularization parameter %s' % (best_rank, best_para)
    return best_rank, best_para

def test(train_set, test_set, rank, reg_para, partitionNum):
    train_set = train_set.repartition(partitionNum).cache()
    test_predict = test_set.map(lambda x : (x[0], x[1])).partitionBy(partitionNum).cache()
    iteration = 10
    model = ALS.train(train_set, rank, iterations = iteration, lambda_ = reg_para)
    prediction = model.predictAll(test_predict).map(lambda x : ((x[0], x[1]), x[2]))  
    pred_rate = test_set.map(lambda x : ((int(x[0]), int(x[1])), float(x[2])))\
                        .join(prediction, numPartitions = partitionNum) 
    error = math.sqrt(pred_rate.map(lambda x : (x[1][0] - x[1][1]) ** 2).mean())
    print 'For testing data the RMSE is %s' % (error)



if __name__ == "__main__":
    sc = SparkContext(appName='Movie Recommandation')
    path = 'ml-latest-small/ratings.csv'
    data = proc_data(path)
    partitionNum = 10
    train_set, validation_set, test_set = data.randomSplit([6, 2, 2], seed = 0L)
    rank, reg_para = train(train_set, validation_set, partitionNum)
    test(train_set, test_set, rank, reg_para, partitionNum)