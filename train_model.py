# -*- coding: utf-8 -*-
import numpy as np
from time import time
import sys,argparse
from pyspark import SparkContext
from operator import add
from pyspark.mllib.recommendation import ALS
import math

def proc_data(path, sc):
    rating_raw = sc.textFile(path)
    header = rating_raw.take(1)[0]
    rating_data = rating_raw.filter(lambda line : line != header).map(lambda s : s.split(","))\
                            .map(lambda x : (x[0], x[1], x[2]))
    return rating_data

def train(train_set, validation_set, rank, reg_para, iteration, partitionNum):
    # train_set, validation_set -> (user, movie, rating)
    train_set = train_set.repartition(partitionNum).cache()
    
    error = 0

    # (user, movie, rating) -> (user, movie)
    validation_predict = validation_set.map(lambda x : (x[0], x[1])).partitionBy(partitionNum).cache()

    # Train model with different parameters
    start = time()
    model = ALS.train(train_set, rank = rank, iterations = iteration, lambda_ = reg_para)
    prediction = model.predictAll(validation_predict).map(lambda x : ((x[0], x[1]), x[2]))

    # ((user, movie), (rating, predict_rating))
    pred_rate = validation_set.map(lambda x : ((int(x[0]), int(x[1])), float(x[2])))\
                              .join(prediction, numPartitions = partitionNum)
    error = math.sqrt(pred_rate.map(lambda x : (x[1][0] - x[1][1]) ** 2).mean())
    now = time() - start

    return rank, reg_para, error, now
            

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description = 'Movie Recommandation.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--data',default=None, help='Input file containing (user, movie, rating, timestamp), used to train a logistic model')
    parser.add_argument('--rank', type=int,default=10, help='number of features to use (also referred to as the number of latent factors)')
    parser.add_argument('--lam', type=float,default=0.01, help='Regularization parameter λ')
    parser.add_argument('--iteration', type=int,default=10, help='Maximum number of iterations for ALS')
    parser.add_argument('--partitionNum',type=int,default=2,help='Level of parallelism')

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=False)

 
    args = parser.parse_args()
    sc = SparkContext(appName='Movie Recommandation')

    if not args.verbose :
        sc.setLogLevel("ERROR")  

    if args.data is not None:
        print 'Reading data and constrcut train, validation and test data from', args.data        
        data = proc_data(args.data, sc).cache()
        train_set, validation_set, test_set = data.randomSplit([6, 2, 2], seed = 0L)
        train_set.saveAsTextFile('small_train')
        validation_set.saveAsTextFile('small_validation')
        test_set.saveAsTextFile('small_test')
    else:
        train_set = sc.textFile('small_train').map(eval)
        test_train = sc.textFile('small_test').map(eval)
        validation_set = sc.textFile('small_test').map(eval)

    print 'Training on data from train set, with λ =',args.lam,', max iter = ',args.iteration, ', rank= ',args.rank
    rank, para, error, now = train(train_set, validation_set, args.rank, args.lam, args.iteration, args.partitionNum)
    print 'For rank %s and regularization parameter %s the RMSE is %s, took %s seconds' % (rank, para, error, now)    
