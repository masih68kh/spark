import sys
import argparse
import numpy as np
#from pyspark import SparkContext

def toLowerCase(s):
    """ Convert a string to lowercase. E.g., 'BaNaNa' becomes 'banana'
    """
    return s.lower()

def stripNonAlpha(s):
    """ Remove non alphabetic characters. E.g. 'B:a,n+a1n$a' becomes 'Banana' """
    return ''.join([c for c in s if c.isalpha()])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Text Analysis through TFIDF computation',formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
    parser.add_argument('mode', help='Mode of operation',choices=['TF','IDF','TFIDF','SIM','TOP']) 
    parser.add_argument('input', help='Input file or list of files.')
    parser.add_argument('output', help='File in which output is stored')
    parser.add_argument('--master',default="local[20]",help="Spark Master")
    parser.add_argument('--idfvalues',type=str,default="idf", help='File/directory containing IDF values. Used in TFIDF mode to compute TFIDF')
    parser.add_argument('--other',type=str,help = 'Score to which input score is to be compared. Used in SIM mode')
    args = parser.parse_args()
   
    sc = SparkContext(args.master, 'Text Analysis')

    if args.mode=='TF' :
        # Read text file at args.input, compute TF of each term, 
        # and store result in file args.output. All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings, i.e., "" 
        # are also removed
         
        bodyText = sc.textFile(args.input)
        words = bodyText.flatMap(lambda lines: lines.split()).map(lambda word: \
                ''.join([c for c in word if (c<='z' and c >='a') or (c<='Z' and c >='A')]).lower())\
                    .filter(lambda wrd: wrd != '')
        words.map(lambda word: (word, 1)).reduceByKey(lambda x,y: x+y).saveAsTextFile(args.output)

    if args.mode=='TOP':
        # Read file at args.input, comprizing strings representing pairs of the form (TERM,VAL), 
        # where TERM is a string and VAL is a numeric value. Find the pairs with the top 20 values,
        # and store result in args.output
        
        freqOfPairs = sc.textFile(args.input)
        freqOfPairs_list = freqOfPairs.map(eval).takeOrdered(20, key=lambda pair: -pair[1])
        # freqOfPairs_list is a list obj contains the most 20 frequnctly used words
        # We are going to save is as a text file

        # creating a string out of a list
        Ostr = ''
        for wrd in freqOfPairs_list:
            Ostr += str(wrd) + '\n'
        # writing the string to a file
        with open(args.output, 'a') as fp:
            fp.write(Ostr)
        
    if args.mode=='IDF':
        # Read list of files from args.input, compute IDF of each term,
        # and store result in file args.output.  All terms are first converted to
        # lowercase, and have non alphabetic characters removed
        # (i.e., 'Ba,Na:Na.123' and 'banana' count as the same term). Empty strings ""
        # are removed
        allFiles = sc.wholeTextFiles(args.input).cache() # rdd with (path, content)
        N = allFiles.count()
        filePath_word = allFiles.flatMapValues(lambda strng: strng.split()).mapValues(lambda word: \
                ''.join([c for c in word if (c<='z' and c >='a') or (c<='Z' and c >='A')]).lower())\
                    .filter(lambda wrd: wrd != '')
        word_filePath = filePath_word.map(lambda pair: (pair[1],pair[0]))
        word_filePath_Union =  word_filePath.combineByKey(lambda x: {x}, lambda x,y: x.union({y}), \
                                                            lambda x,y:x.union(y))
        word_filePath_count = word_filePath_Union.mapValues(lambda set: len(set))
        word_idf = word_filePath_count.mapValues(lambda num: np.log(N/(num+1)))
        word_idf.saveAsTextFile(args.output)

    if args.mode=='TFIDF':
        # Read  TF scores from file args.input the IDF scores from file args.idfvalues,
        # compute TFIDF score, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL),
        # where TERM is a lowercase letter-only string and VAL is a numeric value. 

        TFrdd = sc.textFile(args.input).map(eval)
        IDFrdd = sc.textFile(args.idfvalues).map(eval)
        TFrdd.join(IDFrdd).mapValues(lambda pair: pair[0]*pair[1])\
            .sortBy(lambda pair: pair[1], ascending=False).saveAsTextFile(args.output)
        
    if args.mode=='SIM':
        # Read  scores from file args.input the scores from file args.other,
        # compute the cosine similarity between them, and store it in file args.output. Both input files contain
        # strings representing pairs of the form (TERM,VAL), 
        # where TERM is a lowercase, letter-only string and VAL is a numeric value. 
        
        TFIDF1 = sc.textFile(args.input).map(eval).cache()
        TFIDF2 = sc.textFile(args.other).map(eval).cache()
        nom = TFIDF1.join(TFIDF2).map(lambda pair: pair[1][0]*pair[1][1]).reduce(lambda x,y: x+y)
        denom1 = TFIDF1.map(lambda pair: pair[1]**2).reduce(lambda x,y:x+y)
        denom2 = TFIDF2.map(lambda pair: pair[1]**2).reduce(lambda x,y:x+y)
        denom = np.sqrt(denom1*denom2)
        with open(args.output, 'a') as fp:
            fp.write(str(nom/denom)+"\n")




