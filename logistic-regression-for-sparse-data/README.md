# Logistic Regression using Spark on Sparse Data
Generating a logistic model on a text dataset using pyspark

---

## How to run the app:
run the algorithm by `python LogisticRegression.py`   
make the algorithm runs parallel using Spark by `python ParallelLogisticRegression.py`   

---

## Dataset:   
dataset contains tuples of the for (X,y) in which y ∈ {-1,+1} indicating the class label.  
X is a sparse vector that demonstrates the features that have a value. If a feature does not exist in X, it meants it's zero.   
X is saved as python dict class, but another class called SparseVector is define inhering form dict class that has more atributes and methods.

