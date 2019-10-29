import os
import numpy as np
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
lam_vec = np.linspace(0,10,num=11,dtype=float)
mse = []

for lam in lam_vec:
    proc = subprocess.Popen(["spark-submit --master local[20] --driver-memory 60G\
         ParallelRegression.py --train data/small.train --test data/small.test\
     --beta beta_small_0.0 --lam {} --eps 0.01".format(lam)], stdout=subprocess.PIPE, shell=True)
    (output, err) = proc.communicate()
    output = output.decode("utf-8")
    output = output.split()
    for idx , word in enumerate(output):
        if word == 'MSE' and output[idx+1] == 'is:':
            mse.append( eval(output[idx+2] ))
            break

fig = plt.figure(figsize=(5,5))
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.plot(lam_vec,mse)
fig.savefig('fig_small.png')

mse = []
for lam in lam_vec:
    proc = subprocess.Popen(["spark-submit --master local[20] --driver-memory 60G\
         ParallelRegression.py --train data/big.train --test data/big.test\
     --beta beta_big_best.0 --lam {} --eps 0.6".format(lam)], stdout=subprocess.PIPE, shell=True)
    (output, err) = proc.communicate()
    output = output.decode("utf-8")
    output = output.split()
    for idx , word in enumerate(output):
        if word == 'MSE' and output[idx+1] == 'is:':
            mse.append( eval(output[idx+2] ))
            break

fig = plt.figure(figsize=(5,5))
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.plot(lam_vec,mse)
fig.savefig('fig_big.png')