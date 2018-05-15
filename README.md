# Tweets Clustering using k-means algorithm

Twitter provides a service for posting short messages. In practice, many of the tweets are very similar to each other and can be 
clustered together. By clustering similar tweets together, we can generate a more concise and organized representation of the raw tweets, which will be very
useful for many Twitter-based applications (e.g., truth discovery, trend analysis, search ranking, etc.)
In this project, we have clustered the tweets by utilizing Jaccard Distance metric and K-means clustering algorithm.

**How to run the code?**
-------------------------------

Install the required libraries:

pip3 install nltk

Open python command line and execute the following commands

>import nltk
>nltk.download('stopwords')
-----------------------------------------------------------------------------------------------

Command to run the code:

python tweetKMeansCentroid.py 25 InitialSeeds.txt Tweets.json tweets-k-means-output.txt

(or)

python tweetKMeansCentroid.py Tweets.json tweets-k-means-output.txt

-----------------------------------------------------------------------------------------------
