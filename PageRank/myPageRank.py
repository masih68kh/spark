from pyspark import SparkContext
import time
import os
if os.path.exists("result.txt"):
    os.system('rm -rf result.txt')
    os.system('rm fig.png')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

error_list = []
def add(x,y):
    return x+y
def is_close(rdd1, rdd2, epsilon=1e-5):
    """
    checks is two rdds are close enough
    rdds should be in the form of key-value pair and each elements (int, float)
    """
    error = rdd1.join(rdd2).mapValues(lambda score_tuple: (score_tuple[0]-score_tuple[1])**2)\
        .aggregate(0, lambda flt, pair: flt+pair[1], lambda flt1,flt2: flt1+flt2)
    error_list.append(error)
    return error < epsilon

startTime = time.time()
sc = SparkContext('local[20]',"myPageRank")
originalRDD = sc.textFile('graph').map(eval).mapValues(lambda target: [target])\
                .reduceByKey(lambda x,y: x+y, numPartitions=20).cache()
# originalRDD contains (w, [outDegreeW' ...])
N = originalRDD.count()
scoresRDD = originalRDD.keys().map(lambda w: (w,1./N)).partitionBy(20).cache()
gamma = 0.15

for it in range(40):
    print("********")
    print("interation # %d"%it)
    joined = originalRDD.join(scoresRDD)
    old_scoresRDD = scoresRDD
    scoresRDD = joined.flatMap(lambda pair: [(w,pair[1][1]/len(pair[1][0])) for w in pair[1][0]])\
            .reduceByKey(add, numPartitions=20).mapValues(lambda s: gamma/N + (1-gamma)*s).cache()
    if is_close(scoresRDD, old_scoresRDD, epsilon=-1e-8):
        break
    old_scoresRDD.unpersist()
scoresRDD = scoresRDD.sortBy(lambda pair: -pair[1])
scoresRDD.saveAsTextFile('result.txt')
endTime = time.time()
print("************")
print("time : ", endTime-startTime, " s")
fig = plt.figure(figsize=(5,5))
ax = fig.add_axes((0.2,0.2,0.7,0.7))
ax.plot(error_list)
fig.savefig("fig.png")
