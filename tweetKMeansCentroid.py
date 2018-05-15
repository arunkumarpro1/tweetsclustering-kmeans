import sys
import json
import re, string
import copy
from nltk.corpus import stopwords

regex = re.compile('[%s]' % re.escape(string.punctuation))
cachedStopWords = stopwords.words('english')

#Main class for KMeans
class kMeansClustering():
    def __init__(self, K, centroids, tweets):
        self.centroids = centroids
        self.tweets = tweets
        self.max_iterations = 1000
        self.K = K

        self.clusters = {}
        self.revClusters = {}
        self.distanceMatrix = {} 

        self.setupClusters()
        self.findDistanceMatrix()

    #Routine to calculate JaccardDistance between two tweetd
    def calculateJaccardDistance(self, setA, setB):
            return 1 - float(len(setA.intersection(setB))) / float(len(setA.union(setB)))

    #Routine to get bag of words from tweets
    def getWordsFromTweet(self, string):
        words = string.lower().strip().split(' ')
        for word in words:
            word = word.rstrip().lstrip()
            if not re.match(r'^https?:\/\/.*[\r\n]*', word) \
            and not re.match('^@.*', word) \
            and not re.match('\s', word) \
            and word not in cachedStopWords \
            and word != 'rt' \
            and word != '':
                yield regex.sub('', word)

    #Routine to calculate distance between all pairs of tweets and store in a matrix
    def findDistanceMatrix(self):
        for ID1 in self.tweets:
            self.distanceMatrix[ID1] = {}
            bag1 = set(self.getWordsFromTweet
                (self.tweets[ID1]['text']))
            for ID2 in self.tweets:
                if ID2 not in self.distanceMatrix:
                    self.distanceMatrix[ID2] = {}
                bag2 = set(self.getWordsFromTweet(self.tweets[ID2]['text']))
                distance = self.calculateJaccardDistance(bag1, bag2)
                self.distanceMatrix[ID1][ID2] = distance
                self.distanceMatrix[ID2][ID1] = distance

    #Routine to initialize clusters
    def setupClusters(self):
        for ID in self.tweets:
            self.revClusters[ID] = -1

        for k in range(self.K):
            self.clusters[k] = set([self.centroids[k]])
            self.revClusters[self.centroids[k]] = k

    #Routine to form clusters based on new centroids
    def formNewClusters(self):       
        for ID in self.tweets:
            min_dist = float("inf")
            min_cluster = self.revClusters[ID]

            for k, centroid in enumerate(self.centroids):
                if min_dist > self.distanceMatrix[ID][centroid]:
                    min_dist = self.distanceMatrix[ID][centroid]
                    min_cluster = k
            if self.revClusters[ID] != -1:
                self.clusters[self.revClusters[ID]].remove(ID)
            self.clusters[min_cluster].add(ID)
            self.revClusters[ID] = min_cluster

    #Routine to compute new centroids based on tweets in the cluster
    def computeNewCentroids(self):
        newCentroids = []
        sum_dist = 0
        for k in self.clusters:
            min_dist = float("inf")
            centroid = self.centroids[k]
            for ID1 in self.clusters[k]:
                for ID2 in self.clusters[k]:
                    sum_dist += self.distanceMatrix[ID1][ID2]
                if min_dist > sum_dist:
                    min_dist = sum_dist
                    centroid = ID1
                sum_dist = 0
            newCentroids.append(centroid)
        return newCentroids

    #Routine to perform KMeansClustering
    def performIterationsOrConverge(self):
        self.formNewClusters()
        iterations = 1

        while iterations < self.max_iterations:
            newCentroids = self.computeNewCentroids()
            iterations += 1
            if self.centroids != newCentroids:
                self.centroids = newCentroids
                self.formNewClusters()
            else:
                return

    #Routine to calculate the sum of squared error for all clusters
    def computeSSE(self):
        SSE = 0
        for k in self.clusters:
            for ID in self.clusters[k]:
                SSE += self.distanceMatrix[self.centroids[k]][ID]**2
        return SSE
         
    #Prints the clusters to the given output file   
    def printOutputToFile(self, outputFile):
        with open(outputFile, 'w') as file:
            for k in self.clusters:
                file.write(str(k) + ' ' + ','.join(map(str,self.clusters[k])) + '\n')
            file.write("\nSSE = "+ str(self.computeSSE()))


#Main routine
def main():
    seedsFile = 'InitialSeeds.txt'
    K = 25
    if len(sys.argv) == 5:
        K = int(sys.argv[1])
        seedsFile = sys.argv[2]
        tweetFile = sys.argv[3]
        outputFile = sys.argv[4]
    elif len(sys.argv) == 3:
        tweetFile = sys.argv[1]
        outputFile = sys.argv[2]
    else:
        print('Usage : tweetKMeansCentroid <numberOfClusters> <initialSeedsFile> <TweetsDataFile> <outputFile> (or) tweetKMeansCentroid <TweetsDataFile> <outputFile>')
        exit(-1)
    
    tweetById = {}
    with open(tweetFile, 'r') as f:
        for line in f:
            tweet = json.loads(line)
            tweetById[tweet['id']] = tweet
    
    with open(seedsFile, 'r') as f:
        centroids = [int(line.rstrip(',\n')) for line in f.readlines()]

    handle = kMeansClustering(K, centroids, tweetById)
    handle.performIterationsOrConverge()
    handle.printOutputToFile(outputFile)
    

if __name__ == '__main__':
    main()