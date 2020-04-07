# Plagiarism_Detector
**Key Idea: Quickly detecting plagiarisms between documents using K-Shingling, minHash and LSH techniques.**

**More Info: Having a big corpus/dataset of documets as source, the algorith tries to detect quickly pairs of documents which may be plagiarised.**



### Preprocess
#### At this part of the code, I am preprocessing all texts from the given documents.

**Transforming all the letters in lower case.** - I am doing this because I want the computer to be able to count the words spelled with the same letters as the same words. (ex. “Hello” == “hello”).
**Tokenization** - This means that each word becomes a token.
**Removing Sumbols** - Removing all symbols and numbers from each token.
**Lemmatisation** - In this phase, each word - token is getting lemmatised based on loaded vocabularies. The reason why I didn’t chose stemming instead is because I am looking for plagiarisms and plagiarisms may often happen with totally different words from the same lemma. But the  truth is that in order to check their results, I used them both and the interesting thing is that stemming took me much more time which I was nor expecting to be honest, because I think that lemmatisation is a more sophisticated method since it does not just cut beginnings or endings of words, but instead it also uses vocabularies to find lemmas.

### K - Shingling
So, after reading the text of each document and preprocessing it, I am using the technique of K-Shingling. In other words, for each document I am keeping a set of all possible consecutive substring of length k found within it. I tried different values of k. Like k=3 or k=5 or k=10 or k=20 but there was not any big difference, so I have just chosen k=5 for my calculations. 
It is important to mention that I am also keeping a vocabulary of all shingles used in the corpus.

Usually it is said that the best way to continue is by creating a matrix with as much rows as the unique singles of the corpus and as much columns as the documents. Each item (i,j) of the matrix has a value of 0 or 1 which defines whether or not a shingle i is used in a document j. I am not using this perspective though. I am just keeping a vocabulary with a set of shingles for each document. Before keeping the shingle I am compressing it in 4 bytes to make the processes faster.

### MinHashing
And here comes the interesting part! We are at a point in which we want to start calculating similarities among documents. If we do this sequentially by taking one after the other it will take as a really big amount of time. Thus, we need to find a way that could give us only pairs of documents which have a “big” possibility of being plagiarised.

**Notice:**  *The method I am using for calculating similarity is “Jaccard Similarity”. For any two given sets, their Jaccard Similarity is the size of their intersection divided by their union. This is a decent method for few documents but obviously it is not scaling well when the number of documents increases too much like in our situation.* 

Min Hashing will give us the ability to create small signatures which will represent the shingles of each document. More details about the exact way of creating these signatures are well explained in comments in the Jupyter Notebook tutorial.

The point is that at at the end of this process we will have a signature matrix with as many columns as the documents and as many rows as the random hash functions used.

Since for now on we are going to calculate the Jaccard Similarity of Signatures for each pair of documents and not the Jaccard Similarity of their whole set of Shingles, we have to keep in mind that we are going to just calculate an approximation of their similarity and not their absolute similarity, but the good thing here is that we will need much less time.
Also, it worths to mention that as the number of random hash functions we are using increases, the estimations will be closer to reality.

### Locality Sensitive Hashing (LSH)
So at this point having the Signature matrix is a good thing because we have a “smaller” matrix instead of a big dictionary for the whole corpus, but the problem that we need to compare a lot of pairs of documents still exists.

To solve this, I am partitioning the Signatures Matrix into bands of rows. The product of bands x rows must be equal to the total number of hash functions used of course.

LSH, is going to take all bands of the same rows for each document and hash them into buckets. The amount of buckets should be in a size which helps enough signatures but not too many, to get hashed in the same bucket. So I opted for 211 which is a prime number. At the end of the process for each set of bands, if a bucket has partial signatures of more than one document, then these documents are considered as possible pairs of plagiarised texts and in this case, next step is the calculation of the Jaccard Similarity of their hashed (32bit) Shingles. If the approximation of their similarity is greater than a threshold which we have predefined, then the pair is added in the list of possible plagiarised documents.

Another thing I would like to mention is that I can notice a “trading thing” here. So, having in mind that I am using 150 hash functions, if I take 30 bands with 5 rows each,I can find almost 420/451 correct plagiarised documents but this gives me around 4000 possible pairs which is a big number to check them all one by one. On the other hand, if a take 15 bands with 10 rows each, I only have to check 1700 different pairs of candidate documents for plagiarism but then I will be able to only detect around 370/451 plagiarised files. So, each time it depends on my needs - if I want to be more fast or more accurate with my processes. This time I am choosing to go with the time, so I will keep 15 bands of 10 rows each.

### Searching for plagiarisms
At this point what I have managed to do, is to restrict the number of comparisons between documents from about 900*900 = 810.000 different comparisons to only 1700 by sacrificing some (not to much) plagiarised documents which I will not detect. Of course this is not a serious problem if we consider that doing these 810.000 comparisons would need to run for many many days…
A plain algorithm which searches and prints similar 5-Shingles considering synonyms (please have in mind that this algorithm is not mine - I found something similar on the internet and I just adapted in to my needs) is beginning to check for plagiarisms for each one of the pairs.



**Note:** *In the notebook I have stopped the process after printing all plagiarisms between the first candidate pair, in order to observe the results more efficiently. Of course you can run at anytime on your own the code to see more results. Get prepared though, because it is taking time.*
