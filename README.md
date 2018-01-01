# PASSAT
PASSAT (PAtient Satisfaction Sentiment Analysis Tool)

### The Data
The data comes from a patient satisfaction survey and includes free-text responses to around 40 different questions.  Since the data is PHI ("Personal Health Information"), it is not included here.  However, the code should be easy adjustable to any similar data.

The code cleans the data by removing all stop words and stemming what is left (using a Porter stemmer).

### The Models
The code first uses Latent Dirichlet Allocation to perform "topic modeling" - categorizing the comments into a user-selected number of topics. 

Sentiment analysis for each comment is performed using VADER.

The final result is a "Rotten Tomatoes"-type analysis where each topic is given a rating (the percentage of comments in each topic that are positive).
