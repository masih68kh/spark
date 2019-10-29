# Text Analyzer using Spark
This application calculates different parameters associated with a text corpus.  
The parameters include TF, IDF, TFIDF, and cosine similarity of a text/corpus.

---

## How to run the app:
run the app by `python TextAnalyzer.py [-h] [--master MASTER] [--idfvalues IDFVALUES] [--other OTHER] {TF,IDF,TFIDF,SIM,TOP} input output`   


Text Analysis through TFIDF computation

```console
positional arguments:  
  {TF,IDF,TFIDF,SIM,TOP}  
                        Mode of operation  
  input                 Input file or list of files.  
  output                File in which output is stored  

optional arguments:  
  -h, --help            show this help message and exit  
  --master MASTER       Spark Master (default: local[20])  
  --idfvalues IDFVALUES  
                        File/directory containing IDF values. Used in TFIDF  
                        mode to compute TFIDF (default: idf)  
  --other OTHER         Score to which input score is to be compared. Used in  
                        SIM mode (default: None)  
```


### the data that the operation is done on, is a corpus of text under "masc_500k_texts"
