import os
import numpy as np
import subprocess
import matplotlib
from tqdm import tqdm
matplotlib.use('Agg')
import matplotlib.pyplot as plt
lam_vec = np.linspace(0,10,num=11,dtype=float)
mse = []

for d in tqdm(range(1,11)):
    proc = subprocess.Popen(["spark-submit --master local[20] --driver-memory 100G\
         MFspark.py small_data 5 --lam 0 --mu 0\
        --d {}".format(d)], stdout=subprocess.PIPE, shell=True)
    (output, err) = proc.communicate()
    output = output.decode("utf-8")
    output = output.split()
    for idx , word in enumerate(output):
        if word == 'error' and output[idx+1] == 'is:':
            mse.append( eval(output[idx+2] ))
            break

fig = plt.figure(figsize=(5,5))
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.plot(list(range(1,11)),mse)
fig.savefig('d.png')

lam_vec = np.linspace(0,50,num=20,dtype=float)
for par in tqdm(lam_vec):
    proc = subprocess.Popen(["spark-submit --master local[20] --driver-memory 100G\
         MFspark.py small_data 5 --lam {} --mu {}\
        --d 10".format(par,par)], stdout=subprocess.PIPE, shell=True)
    (output, err) = proc.communicate()
    output = output.decode("utf-8")
    output = output.split()
    for idx , word in enumerate(output):
        if word == 'error' and output[idx+1] == 'is:':
            mse.append( eval(output[idx+2] ))
            break

fig = plt.figure(figsize=(5,5))
ax = fig.add_axes([0.1,0.1,0.8,0.8])
ax.plot(lam_vec,mse)
fig.savefig('mu_lambda.png')