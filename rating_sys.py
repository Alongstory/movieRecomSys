from pyspark.mllib.recommendation import ALS
from pyspark import SparkContext
from time import time
import sys,argparse
import math

def proc_data(path, sc):
	'''
	process data for computing

	'''
	data_raw = sc.textFile(path)
	header = data_raw.take(1)[0]
    data = data_raw.filter(lambda line : line != header).map(lambda s : s.split(","))\
                   .map(lambda x : (x[0], x[1], x[2]))
	return data

def cal_RMSE(train_set, test_set, partitionNum, iteration):
	'''
	calculate RMSE for test dataset of complete data, which can be compared with RMSE of small data
	train_set is used for building model, whose form is (user, movie, rating)
	test_predict is used for testing and calculating RMSE, whose form is (user, movie)
	pred_rate's form is ((user, movie),(real rating, predicted rating))

	'''
	train_set = train_set.repartition(partitionNum)
	test_predict = test_set.map(lambda x : (x[0], x[1])).partitionBy(partitionNum).cache()
	print 'Building and test model ...'
	model = ALS.train(train_set, 4, iterations = iteration, lambda_ = 0.1)
	start = time()
	prediction = model.predictAll(test_predict).map(lambda x : ((x[0], x[1]), x[2]))
	pred_rate = test_set.map(lambda x: ((int(x[0]), int(x[1])), float(x[2]))).join(prediction, numPartitions = partitionNum)
	error = math.sqrt(pred_rate.map(lambda x: (x[1][0] - x[1][1]) ** 2).mean())
	now = time() - start
	return error, now, partitionNum

def add_new_pred(ratings, movies, new_user_data, sc, iteration):
	'''
	Core part I of recommedation, a recommedation sys is to recommend top movies to new users according to his imcomplete existing rating
	This part add new info, build model and predict those movieID that new user haven't rated

	'''
	data_raw = sc.textFile(new_user_data)
	# Get new user's rating info, form:(userID, movieId, rating)
	new_data = data_raw.map(lambda s : s.split(',')).map(lambda x : (int(x[0]), int(x[1]), float(x[2])))
	# Get userID
	new_user = new_data.take(1)[0][0]
	# Add new info to original ratings, expand dataset and build new model
	new_ratings= ratings.union(new_data)
	new_model = ALS.train(new_ratings, 4, iterations = iteration, lambda_ = 0.1)
	# Get the rated movieIDs list
	new_rated_movies_RDD = new_data.map(lambda x : x[1])
	# Convert to list
	new_rated_movies = new_rated_movies_RDD.take(new_rated_movies_RDD.count()) 
	# Get the unrated movieIDs, form:(userId, movieId)
	new_unrated_movies = movies.filter(lambda x : int(x[0]) not in new_rated_movies).map(lambda x : (new_user, x[0]))
	# (movieId, rating)
	prediction = new_model.predictAll(new_unrated_movies).map(lambda x : (x[1], x[2]))
	return prediction

def get_top_recommendation(prediction, movies_titles, partitionNum):
	#(movieId, rating, movieName)
	recommendation = prediction.join(movies_titles, numPartitions = partitionNum).map(lambda (x, y) : (x, y[0], y[1]))
	top_10 = recommendation.takeOrdered(10, lambda x : -x[1])
	return top_10
	


if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description = 'Movie Recommandation.',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--ratings',default=None, help='Input file containing (user, movie, rating, timestamp), used to train a logistic model')
    parser.add_argument('--movies',default=None, help='Input file containing (movieID, movieName, genre), used to train a logistic model')
    parser.add_argument('--newuser',default=None, help='Input file containing (user, movie, rating), used to get recommendation for new user')
    parser.add_argument('--iteration', type=int,default=10, help='Maximum number of iterations for ALS')
    parser.add_argument('--partitionNum',type=int,default=2,help='Level of parallelism')

    verbosity_group = parser.add_mutually_exclusive_group(required=False)
    verbosity_group.add_argument('--verbose', dest='verbose', action='store_true')
    verbosity_group.add_argument('--silent', dest='verbose', action='store_false')
    parser.set_defaults(verbose=False)

 
    args = parser.parse_args()
    sc = SparkContext(appName='Movie Recommandation')

    # Trained paramaters
    rank = 4
    reg_para = 0.1

    if not args.verbose :
        sc.setLogLevel("ERROR")  

    if args.ratings is not None:
    	print 'Construct model from', args.ratings
        ratings = proc_data(args.ratings, sc).cache()
    	train_set, test_set = ratings.randomSplit([7, 3], seed = 0L)

    error, now, partitionNum = cal_RMSE(train_set, test_set, args.partitionNum, args.iteration)
    print 'For partition number %s the RMSE is %s, took %s seconds' % (partitionNum, error, now) 


    if args.movies is not None:
    	print 'Reading movies data from', args.movies
    	movies = proc_data(args.movies, sc).cache()
    	# Form (movieId, movieName)
    	movies_titles = movies.map(lambda x : (int(x[0]), x[1])).cache()
    print 'Building recommendation system...'
    start = time()
    prediction = add_new_pred(ratings, movies, args.newuser, sc, args.iteration)
    top_10 = get_top_recommendation(prediction, movies_titles, args.partitionNum)
    time = time() - start
    print 'Top 10 recommended movies for new user:\n%s' % '\n'.join(map(str, top_10))
    print 'Cost time : %s' % time
