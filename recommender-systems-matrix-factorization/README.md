# Matrix Factorizarion for Recommended Systems 
Matrix Factorizarion for Recommended Systems is developed here using spark (pyspark)

---

## How to run the app:
run the app by `python MFspark.py `  
```console
positional arguments:
  data                  Directory containing folds. The folds should be named
                        fold0, fold1, ..., foldK.
  folds                 Number of folds

optional arguments:
  -h, --help            show this help message and exit
  --gain GAIN           Gain (default: 0.001)
  --power POWER         Gain Exponent (default: 0.2)
  --epsilon EPSILON     Desired objective accuracy (default: 1e-99)
  --lam LAM             Regularization parameter for user features (default:
                        1.0)
  --mu MU               Regularization parameter for item features (default:
                        1.0)
  --d D                 Number of latent features (default: 10)
  --outputfile OUTPUTFILE
                        Output file (default: None)
  --maxiter MAXITER     Maximum number of iterations (default: 20)
  --N N                 Parallelization Level (default: 20)
  --seed SEED           Seed used in random number generator (default:
                        1234567)
  --output OUTPUT       If not None, cross validation is skipped, and U,V are
                        trained over entire dataset and store it in files
                        output_U and output_V (default: None)
  --verbose
  --silent
```

## Hypterparameter Selection
run the script `python run.py` for hyperparameter selection.

---

## dataset
There are two datasets provided for this algorithm:
- small_data, which contains a synthetic data to check the correctness of the code 
- big_data, which contains a subset of the MovieLens dataset (https://grouplens.org/datasets/movielens/)
