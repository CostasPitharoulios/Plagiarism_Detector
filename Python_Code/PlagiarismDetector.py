# =============================================================================
#                 *** Imports & General ***
# =============================================================================
from __future__ import division
import os
import re
import random
import glob
import time
import codecs
import binascii


from bisect import bisect_right
from heapq import heappop, heappush
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag as pos_tagger
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from operator import itemgetter
from nltk.corpus import stopwords as stopwords_nltk, wordnet


import string
import sys
import json
import itertools
from distutils.util import strtobool

srcdir = "corpus"

src_dir_files = [f for f in glob.glob(srcdir + "/" + "*.txt")]

    
stemmer = PorterStemmer()

#################################################################################
#                 *** Reading - Preprocessing - Shingling ***
#################################################################################



# -------------------------------------------------------------------------
#            *** Preprocessing Text ***
# -------------------------------------------------------------------------
# Preprocessing the text of a document 
# Lower case of all words, stemming, removing symbols and numbers
# -------------------------------------------------------------------------
def preprocess_text(fid):
    words = [word.lower() for line in fid for word in line.split()] 
    
    SYMBOLS = '\'\"_{}()[].,:;+-*/&|<>=~$1234567890'
    clear_words = [item.translate(str.maketrans('','',SYMBOLS)).strip() for item in words]
    clear_words = filter(None, clear_words)

    lemmatized_words = []
    for w in clear_words:
        lemmatized_words.append(lemmatizer.lemmatize(w))
        
    
    return(lemmatized_words)
# -------------------------------------------------------------------------
    
    

# =============================================================================
#       *** Converting documents to sets of Shingles ***
# -----------------------------------------------------------------------------
# In this part we are converting documents to sets of shingles.
# We are iterating through each file of the corpus reading it.
# We are preprocessing their text by stemmening and removing symbols and then
# we are creating k-shingles with these words.
# We are keeping them in sets() which means that there are no dublicates!
# =============================================================================


t0 = time.time()

print("Just started reading each file and k-shingling...")


# We use this variable in order to keep track of the Shingles.
# Whenever a new shingle is added, we increment this by one.
#curShingleID = 0

# In this list we are going to keep the names of all document of the corpus.
docNames_list = []

#docsAsShingleSets = {};
  

# This keeps the total number of documets inside the corpus.
numDocs = 0  
# Creating a dictionary which keeps all the Shingle sets of each document of the corpus.
# The key of a document in this dictionary is its id.
corpusShingles = {}

# Keeps the number of documents we have read.
# We will use this just for printing information during the process.
counter = 0;
# For each document inside the corpus
for src_file in src_dir_files:  

    counter += 1 
    
    # Open the document
    fid = open(src_file, newline='')
    
    # Preprocess document's text (stemming, removing symbols & numbers, etc.)
    words = preprocess_text(fid)
    
    numDocs += 1
  
    # Keeping the name of the document  
    docName = src_file.split("/")[1]
  
    # Adding the name of the document in the list of all document names of the corpus  
    docNames_list.append(docName)
    
    
    # This is a set which contains all k-Shingles created from the current document.
    # In fact, each item of the set contains a hash value created with k-Shingles.
    # There are no dublicates of k-Shingles since we have a set
    docShingles = set()

  
    # For each word of the text
    for indWord in range(0, len(words) - 2):
        
        # Creating a Shingle by taking words in a raw from the text.
        shingle = words[indWord] + " " + words[indWord + 1] + " " + words[indWord+ 2] 

        
        # Now we need to compress the Shingle.
        # To compress the Shingle, we hash it to 4 bytes (32 bits).
        hashedShingle = binascii.crc32(str.encode(shingle)) & 0xffffffff
    
        # Adding the hashed Shingle in the set of k-Shingles of the document.
        # If this specific Shingle already exists, then it won't get inserted again.
        docShingles.add(hashedShingle)
  
    # Storing the set of Shingles of this document to the dictionary of Shingles of the whole corpus.
    corpusShingles[docName] =  docShingles
    
    # Closing file
    fid.close()  
    
    if counter % 100 == 0:
        print("Finished reading: " + str(counter) + "/902")
        
 
    # ===========================================================================
# Print total time need for reading, preprocessing and shingling all corpus documents.
print("\n\nREADY. Total time needed for reading, preprocessing, and shingling " +  str(numDocs) + str(" documents: %.2f seconds.") % (time.time() - t0))

 
#print '\nAverage shingles per doc: %.2f' % (totalShingles / numDocs)

#################################################################################
#                 *** Creating MinHash Signatures ***
#################################################################################


# -------------------------------------------------------------------------
#       *** Creating random variables for the hashing functions ***
# -------------------------------------------------------------------------
def createVariables(numHashes):
    
    maxShingleID = 2**32-1 # 2^32-1 is the max hashed Shingle id we can have

    # List which contains all random variables thar are going to be created.
    variables = []
  
    # Create a random variable for each one of the hash functions
    for i in range(numHashes):
        randomVariable = random.randint(0, maxShingleID) 
        
        # Variable created must be unique.
        while randomVariable in variables: 
            randomVariable = random.randint(0, maxShingleID) 
    
        # Appending new variable to the list of random variables.
        variables.append(randomVariable)
    
    return variables

# -------------------------------------------------------------------------
#            *** Creating random hashing functions ***
# -------------------------------------------------------------------------
def randHashFunc_gen(hashnum):
   
    # Keeping the number of hash functions.
    numHashes = hashnum;
    
    # Create a and b variables for hash functions.
    As= createVariables(numHashes)
    Bs = createVariables(numHashes)
    
    signatureMatrix = MinHashSig_gen(numHashes, As, Bs)
    return signatureMatrix



# =============================================================================
#       *** Creating random hash functions ***
# -----------------------------------------------------------------------------
# The hash function we are going to create follo this rule:
#              h(x) = (a*x + b) mod c
# Where a,b are random variables and c is a prime number greater than 
# the maximun value of a hashed Shingle (shingleID)
# =============================================================================


def MinHashSig_gen(numHashes, As, Bs):

    print("Just started creating signatures...")

    
    # This is a list of lists or lists of signatures.
    # Each row represents a document from the corpus and
    # each column represents the minhash value.
    signatureMatrix = []

   
    #-------------------------------------------- 
    # Now it is the time for minshashing...
    # For each of the corpus' documents we are going to hash all its compressed k-Shingles
    # and finally keep the lowest value.
    #-------------------------------------------- 
    
    # This is the largest prime number after the maximun id of Shingle 
    # that we are going to assign.
    nextLargestPrime = 4294967311
    
    
    # Keeps the number of documents we have read.
    # We will use this just for printing information during the process.
    counter = 0;
    
    # For each corpus' document
    for docName in docNames_list:

        counter += 1
        
        # This is gonna keep the signature we are creating for the doc.
        signature = []
        
        # Getting the set with all Shingles of the document.
        setShingle = corpusShingles[docName]

        # For each one of the random hash functions
        for i in range(0, numHashes):
            minHashValue = float("inf")
            # For each shingle in the document
            for shingleID in setShingle:
                
                # Creating hash function ( h(x) = (a*x + b) mod c )
                hashFuncValue = (As[i] * shingleID + Bs[i]) % nextLargestPrime
                # Keeping the lowest value
                if hashFuncValue < minHashValue:
                    minHashValue = hashFuncValue

            # Add the smallest hash code value as component number 'i' of the signature.
            
            # Appending the min hashing value in document's signature.
            signature.append(minHashValue)

        # Store the MinHash signature for this document.
        signatureMatrix.append(signature)
        
        # Prints information about the progress
        if counter % 100 == 0:
            print("Finished creating signatures of: " + str(counter) + "/" + str(numDocs))
    
    return signatureMatrix


# =============================================================================
# Time this step.
t0 = time.time()

signatureMatrix = randHashFunc_gen(150)

# Calculate the elapsed time (in seconds)
elapsed = (time.time() - t0)
        
print( "\n\nREADY. Total time needed for MinHash signatures: %.2fsec" % elapsed)

#################################################################################
#                 *** LSH ***
#################################################################################


# -------------------------------------------------------------------------
#            *** Creating a list of buckets for all bands ***
# -------------------------------------------------------------------------
# Creating a list of lists. 
# Each item of the list is like a pointer to the bucket of each band.
# The list of each one of these items, represents the buckets.
# -------------------------------------------------------------------------
def create_listOfBuckets(bands, bucketLength):
    listOfAllBuckets = []
    for aBand in range(bands):
        #for index in range(bucketLength):
           # listOfAllBuckets.append([])
        listOfAllBuckets.append([[] for i in range(bucketLength)])
    return listOfAllBuckets


# -------------------------------------------------------------------------
#            *** Creating random hashing functions ***
# -------------------------------------------------------------------------
def jaccard_similarity(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection) / union

# =============================================================================
#       *** Applying LSH  ***
# -----------------------------------------------------------------------------
# Separate Signatures Matrix in bands and rows. Each band can have multiple rows.
# It must be: Number of Hash Functions = bands * rows
# The idea here is that we are putting bands of documents in buckets by hashing.
# If two documents for any of their bands (which represent the same rows)
# fall in the same bucket, then we consider them as csnditares for being plagiarised.
# In this case we calculate their Jaccard similarity. If this is greater than
# a threshold we have already defined, then we finally consider them plagiarised.
# =============================================================================

def LSH_func(signatureMatrix,t, b, r):
   
    
    bands = b
    rows = r
    
    # It must always be: Number of Hash Functions = bands * rows.
    if bands * rows != len(signatureMatrix[0]):
        raise "ERROR: Bands * rows are not equal to number of hash function."
   
    # Creating a bucket list for all bands.
    listOfAllBuckets = create_listOfBuckets(bands,211)

    # This is a dictionary which contains all possible pairs of plagiarised documents
    # and their Jaccard Similarity.
    candidates = {}
    
    i = 0
    # For each of the bands
    for aBand in range(bands):
        # Taking the buckets of this specific band
        buckets = listOfAllBuckets[aBand]        
        band = [ele[i:i+rows] for ele in signatureMatrix]
        
        # For each of the partial signatures of this specific band,
        # hash it, in a bucket.
        for row in range(len(band)):
                     
            key = int(sum(band[row][:]) % len(buckets))
            buckets[key].append(row)
        i = i + rows

        # For each part of the bucket
        for item in buckets:
            
            # If there are more than one hashed in the shame bucket
            if len(item) > 1:
                pairNames = (docNames_list[item[0]], docNames_list[item[1]])
                pair = (item[0], item[1])
                
                # If the pair of possible plagiarised documents
                # ther is not already in the list of all candidates
                if pair not in candidates:
                    
                    # We are calculating their Jaccard Similarity
                    #doc1 = corpusShingles[docNames_list[item[0]]]
                    #doc2 = corpusShingles[docNames_list[item[1]]]
                    
                    
                    #Jsim = (len(doc1.intersection(doc2)) / len(doc1.union(doc2)))
                    
                    Jsim = jaccard_similarity(corpusShingles[docNames_list[item[0]]],corpusShingles[docNames_list[item[1]]])
                    
                    # If the Jaccard Similarity of the candidate pair is greater than
                    # the defined threshold, we add them in the dictionary of found plagiarised
                    # documents.
                    if Jsim >= t:
                        candidates[pairNames] = Jsim
                        


   
    sortedCandidates = sorted(candidates.items(),key=itemgetter(1), reverse=True)

    return candidates,sortedCandidates
# =============================================================================

# Time this step.
t0 = time.time()
candidates,sortedCandidates = LSH_func(signatureMatrix,0.001, 15,10)


# Calculating elapsed time
elapsed = (time.time() - t0)
print("\n\nREADY. Total time needed for LSH: %.2fsec" % elapsed)

################################################################################
#                 *** STATISTICS ***
#################################################################################
        



def get_Statistics():

    # Creating a panda dataframe for the file with the labels of documents
    df = pd.read_csv("ground_truth.tsv", sep='\t', usecols = ['doc_name','plagiarism'])
    total = df['plagiarism'].sum()
    print ("The total number of plagiarised documents in the ground_truth.tsv file is: " + str(total))



    # reading all files from corpus and keeping their names
    data = []
    for file in sorted(os.listdir(srcdir)):
        data.append(file)
    df_file = pd.DataFrame(data, columns=['File'])
    df_file = df_file[~df_file.File.str.contains(".xml")]
    df_file['File'] = df_file['File'].str.replace('source-', '')
    df_file['File'] = df_file['File'].str.replace('.txt', '')

    #--------------------------------------------------------------------------------------
    # Now, we want to keep the values 0 or 1 for only the documents which 
    # are included in our corpus.
    # So we are merging existing files in the corpus and all files we know their plagiarism
    # in order to keep the conjuction of them.
    #--------------------------------------------------------------------------------------
    df_merge=df.merge(df_file, left_on='doc_name', right_on='File', how='outer')

    # If a row has a NaN value, means that the file included in this row
    # does not belong to the corpus files, so we are removing it.
    df_merge = df_merge.dropna()


    #print(sort)



    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Processing plagiarised pairs found
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


    # Creating a panda dataframe from all the plagiarized documents 
    # and their similarities.
    df_similars = pd.DataFrame.from_records(sortedCandidates)
    df_similars.columns = ['Pair', 'Jaccard_Sim']

    # Giving names to columns
    df_similars['Document_1'], df_similars['Document_2'] = df_similars.Pair.str

    # Deleting first column
    del df_similars['Pair']

    # Changing the order of columns
    df_similars = df_similars[['Document_1', 'Document_2','Jaccard_Sim']]
    df_allSimilars = df_similars

    # Removing .txt and source- from the file names
    df_similars['Document_1'] = df_similars['Document_1'].str.replace('source-', '')
    df_similars['Document_2'] = df_similars['Document_2'].str.replace('source-', '')
    df_similars['Document_1'] = df_similars['Document_1'].str.replace('.txt', '')
    df_similars['Document_2'] = df_similars['Document_2'].str.replace('.txt', '')
    print("\n\n\n")
    print("======================================================================================")
    print("These are all similar pairs found and their Jaccard Similarity.\n")
    print("======================================================================================")
    print(df_similars) 
    print("\n\n\n")


    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    # Merging all columns
    # By doing this we will be able to find out very easily if a document which 
    # was found as plagiarised, is indeed plagiarized or not.
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    #df_allPlags = pd.DataFrame(columns= ['Documents'])


    # Creating a list of all rows of the dataframe which contains
    # all the similar documents found.
    similars_set = set()  
    # Iterate over each row 
    for index, rows in df_similars.iterrows(): 
        # Create list for the current row 
        doc1 =rows.Document_1
        similars_set.add(doc1)

        doc2 = rows.Document_2
        similars_set.add(doc2)

    #print(len(similars_set))


    # Converting set of plagiarized documents to dataframe
    df_allPlags = pd.DataFrame(list(similars_set), columns= ['Documents'])


    # Merging
    df_allPlags=df.merge(df_allPlags, left_on='doc_name', right_on='Documents', how='outer')
    df_allPlags = df_allPlags.dropna()
    #print("======================================================================================")
    #print("Check if plagiarized documents are indeed plagiarized according to ground_truth.csv\n")
    #print("======================================================================================")
    #print(df_allPlags.to_string())


    print("\n\n\n")

    # Printing the total number of plagiarized documents in our corpus
    # according to ground_truth file.
    total = df_merge['plagiarism'].sum()
    print ("The total number of plagiarised documents is: " + str(total))

    # Printing the total number of plagiarised documents we found.
    print("Total number of plagiarized documents found is: " + str(len(df_allPlags)))
    


    #-------------------------------------------------------
    # Here we are going to calculate how many of the real 
    # plagiarized documents we found
    #-------------------------------------------------------

    #df_allPlags is data frame which contains all the 
    # plagiarised documents found and the value 0 or 1
    # which says the truth. So we are just goind to sum
    # all the correct guesses.

    correct = df_allPlags.loc[df['plagiarism'] == 1, 'plagiarism'].sum()

    # Printing the total number of correct guesses out of the whole number of guesses
    print("The correct guesses are: " + str(correct) + "/" + str(len(df_allPlags)))

    # Printing the total number of correct guesses out of the real number of plagiarised documets
    print("The correct guesses out of all plagiarised documents are: " + str(correct) + "/" + str(total))
    print("\nFalse Positives: " + str((len(df_allPlags) - correct)))
    print("False Negatives: " + str(total - correct))
    
    return df_allSimilars

df_allSimilars = get_Statistics()

################################################################################
#                 *** DETECTING PLAGIARISMS IN CANDIDATE PAIRS ***
#################################################################################

class detect_Plagiarism():

    def __init__(self, doc1, doc2):
    
        self.doc1_text = doc1
        self.doc2_text = doc2
        self.k = 5
        self.synonyms = json.load(open('node_modules/synonyms/src.json', 'r'))
        self.preprocess_Text()
        self.detect_plagShingles()

    def preprocess_Text(self):
        
        
        replace = str.maketrans('', '', string.punctuation)
        
        # Removing punctuation from document 1
        self.doc1_words = self.doc1_text.translate(replace)
        # This is a list which keeps all words of document 1
        self.doc1_words = self.doc1_words.split()
     

        # Removing punctuation document 2
        self.mlist = self.doc2_text.translate(replace)
        # This is a list which keeps all words of document 2
        self.mlist = self.mlist.split()
      
        
        
    def find_Synonyms(self):
        self.mutator_list = []
        for pos, word in enumerate(self.aShingle):
            
            if word in self.synonyms.keys():
                
                syns_list = []

                for key, item in self.synonyms[word].items():
                    syns_list += item

                self.mutator_list.append(list(set(syns_list)))
                    
            else:
                self.mutator_list.append([word])
            
        self.master_list = list(itertools.product(*self.mutator_list))
        
        for aList in self.master_list:
            result = any(list(aList) == self.mlist[it:self.k +it] for it in range(len(self.mlist) -1))
            if result:
                print ("Plagiarism Found in 5-Shingle: ", list(aList))
               

    def detect_plagShingles(self):
       
        print ("Looking for plagiarisms...")
       
        for i,j in enumerate(self.doc1_words):
            self.master_list = []
            self.aShingle = self.doc1_words[i:self.k+i]
            
            # If the length of shingle is less than k continue
            if len(self.aShingle) < self.k:
                continue
            # If the shingle is not yet contained in the list of shingle
            if self.aShingle not in self.master_list:
                self.master_list.append(self.aShingle)
            
            # Check for synonyms
            self.find_Synonyms()
           


for index, rows in df_allSimilars.iterrows(): 
    doc1 = str(srcdir) + "/source-" + rows.Document_1 + ".txt"
    doc2 = str(srcdir) + "/source-" + rows.Document_2 + ".txt"
    print("Searcing for plagiarisms in documents: " + str(rows.Document_1) + " - " + str(rows.Document_2))
    print("------------------------------------------------------------------------")
    quick = detect_Plagiarism(open(doc1, 'r').read(), open(doc2, 'r').read())
    print("\n\n\n\n")