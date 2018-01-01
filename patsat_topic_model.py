#!/usr/bin/env python3

# import libraries and packages
import re
import time
import sys
import numpy as np
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import gensim
from gensim import corpora, models

import warnings
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation

from pyspin.spin import make_spin, Default

# function to import and clean patient survey comments
@make_spin(Default, "Importing and cleaning data...")
def get_clean_comments(topic="All"):

	global train_data_list, test_data_list, orig_train_data, orig_test_data

	if topic == 'All':
		data = survey_data[['Comment']]
	else:
		data = survey_data[survey_data['Question Name'] == topic]
		data = data[['Comment']]

	# ensure comment is a string
	data['Comment'] = data['Comment'].astype(str)

	# shuffle data and split into train/test sets
	data = data.sample(frac=1., random_state=42)

	split = int(np.floor(len(data) * 0.9))
	train_data = data[:split]
	test_data = data[split:]

	# preserve original test data for evaluation at end
	orig_train_data = train_data.copy()
	orig_test_data = test_data.copy()
	orig_test_data.reset_index(inplace=True, drop=True)

	# lowercase and remove any non-alphabetic characters
	train_data['Comment'] = train_data['Comment'].apply(lambda x: word_tokenize(re.sub('[^a-z]', ' ', x.lower())))
	test_data['Comment'] = test_data['Comment'].apply(lambda x: word_tokenize(re.sub('[^a-z]', ' ', x.lower())))

	# remove stopwords
	stop_words = stopwords.words("english")
	stop_words += ['good', 'great', 'excellent', 'dr', 'would', 'use', 'like', 'best', 'always',
	               'could', 'see', 'far', 'much', 'better', 'n', 'awesome', 'love', 'yet', 'well',
	               'every', 'get', 'scca', 'back', 'also', 'one', 'seattle', 'cancer', 'center',
	               'none', 'amazing', 'done', 'wonderful', 'say', 'things', 'enough', 'apply', 
	               'applicable', 'everyone', 'made', 'makes', 'really', 'place', 'name', 'thank',
	               'need', 'na', 'us', 'even', 'outstanding', 'everything', 'bad', 'nice', 'kind',
	               'friendly', 'helpful', 'anyone', 'caring', 'professional', 'went', 'feel',
	               'many', 'fantastic', 'job', 'pa', 'know', 'go', 'way', 'ever', 'want', 'give',
	               'pleasant', 'courteous', 'got', 'gone', 'goes', 'nan']
	train_data['Comment'] = train_data['Comment'].apply(lambda x: [i for i in x if i not in stop_words])
	test_data['Comment'] = test_data['Comment'].apply(lambda x: [i for i in x if i not in stop_words])

	# stem the remaining words
	p_stemmer = PorterStemmer()
	train_data['Comment'] = train_data['Comment'].apply(lambda x: [p_stemmer.stem(i) for i in x])
	train_data['Comment'] = train_data['Comment'].apply(lambda x: [i for i in x if len(i) > 1])
	train_data['Comment'] = train_data['Comment'].apply(lambda x: ' '.join(x))
	train_data.reset_index(inplace=True, drop=True)

	test_data['Comment'] = test_data['Comment'].apply(lambda x: [p_stemmer.stem(i) for i in x])
	test_data['Comment'] = test_data['Comment'].apply(lambda x: [i for i in x if len(i) > 1])
	test_data['Comment'] = test_data['Comment'].apply(lambda x: ' '.join(x))
	test_data.reset_index(inplace=True, drop=True)

	
	train_data_list = train_data['Comment'].tolist()
	test_data_list = test_data['Comment'].tolist()

	print("\n\tNumber of training comments:\t", len(train_data))
	print("\tNumber of test comments:\t ", len(test_data))
	print()

	return train_data_list, test_data_list, orig_test_data

@make_spin(Default, "Training model...")
def topic_train(num_topics):

	global tfidf_vectorizer, lda

	# number of Bag of Words features
	num_features = 1000

	# LDA must use Raw Counts of words
	tfidf_vectorizer = CountVectorizer(ngram_range=(1, 2), 
	                               	   max_df=0.95, 
	                                   min_df=2, 
	                                   max_features=num_features)
	tfidf = tfidf_vectorizer.fit_transform(train_data_list)
	tfidf_feat_names = tfidf_vectorizer.get_feature_names()

	# function to print out topics 
	def display_topics(model, feature_names, num_top_words, train_data_df):
	    
	    for topic_idx, topic in enumerate(model.components_):
	        print("Topic {}:".format(topic_idx))
	        print("---------")
	        print("Top words/phrases: " + ", ".join([feature_names[i] for i in topic.argsort()[:-num_top_words-1:-1]]))
	        print()
	        topic_df = train_data_df[train_data_df['main topic'] == topic_idx]
	        print('Example comment: "' + topic_df.sample(n=1)['Comment'].tolist()[0] + '"')
	        print()


	lda = LatentDirichletAllocation(n_components=num_topics, random_state=1, max_iter=5, 
                                    learning_method='online', learning_offset=50.).fit(tfidf)

	train_data = tfidf_vectorizer.transform(train_data_list)
	train_data_df = pd.DataFrame(lda.transform(train_data), columns=range(num_topics))
	train_data_df = pd.merge(orig_train_data, train_data_df, how='inner', left_index=True, right_index=True)
	train_data_df['main topic'] = train_data_df[np.arange(num_topics)].idxmax(axis=1)
	train_data_df = train_data_df[['Comment', 'main topic']]

	# number of top words to display
	num_top_words = 10

	print("\n\n\nModeled topics with words most associated with them and an example comment: ")
	print("===========================================================================\n ")

	display_topics(lda, tfidf_feat_names, 10, train_data_df)



@make_spin(Default, "Applying model to test data...")
def apply_model():

	global test_data_df, topic_dict

	test_data = tfidf_vectorizer.transform(test_data_list)

	test_data_df = pd.DataFrame(lda.transform(test_data), columns=range(num_topics))

	test_data_df = pd.merge(orig_test_data, test_data_df, how='inner', left_index=True, right_index=True)

	test_data_df['main topic'] = test_data_df[np.arange(num_topics)].idxmax(axis=1)

	test_data_df = test_data_df[['Comment', 'main topic']]
	# counts of topics
	topic_dict = dict(Counter(test_data_df['main topic']))
	

	
@make_spin(Default, "Performing sentiment analysis...")
def sentiment_analysis():

	def vader_analysis(x):

		sid = SentimentIntensityAnalyzer()

		ss = sid.polarity_scores(x)

		return ss['compound']

	test_data_df['sentiment'] = test_data_df['Comment'].apply(vader_analysis)



#@make_spin(Default, "Performing Rotten Tomatoes analysis...")
def rotten_tomato_analysis():

	print()
	print("\nRESULTS: For each topic, we show the number of responses for that topic and the " +\
		           "percentage of POSITIVE comments.")
	print()
	def rotten_tomato_score(topic):
		temp_df = test_data_df[test_data_df['main topic'] == topic]
		temp_df = temp_df[['sentiment']]
		temp_df['positive sentiment'] = temp_df['sentiment'] >= 0.0

		return round(sum(temp_df['positive sentiment']) / len(temp_df) * 100., 1)

	print()
	print("Topic \t\t Number of Reviews \t Pct. Pos. Reviews")
	print("----- \t\t ----------------- \t -----------------")

	for topic in np.arange(num_topics):
		print(str(topic) + ": \t\t\t" + str(topic_dict[topic]) + "\t\t\t" + str(rotten_tomato_score(topic)))




# read in the data
survey_data = pd.read_csv("surveyCommentData.csv", encoding='iso-8859-1')
topic_drop_list = ['If other: ___________ (specify)', 'Additional Comments', 'Uncategorized Comments']

survey_data = survey_data[~survey_data['Question Name'].isin(topic_drop_list)]

# some housekeeping on the question names
survey_data['Question Name'] = survey_data['Question Name'].apply(lambda x: re.sub(' Open-ended', '', x))
survey_data['Question Name'] = survey_data['Question Name'].apply(lambda x: re.sub('Nurses', 'Nursing', x))
questions = dict(Counter(survey_data['Question Name']))
question_list = list(questions.keys())
question_list.append('All')

print("Survey questions: ")
print("================= ")
for i, question in enumerate(question_list):
	print("(" + str(i) + ") " + question) 		# + ": " + str(questions[question]))

print()
print()
while True:
	user_choice = int(input("Which topic would you like to explore? ").strip())
	if user_choice not in range(len(question_list)):
		print("Try again.")
		user_choice = input("Which topic would you like to explore? ").strip()
	else:
		break

user_topic = question_list[user_choice]

train_list, test_list, orig_test = get_clean_comments(user_topic)

while True:
	num_topics = int(input("How many topics would you like to try? [Enter 0 to Exit] ").strip())

	if num_topics == 0:
		break

	topic_train(num_topics)

	print()
	topic_satis = input("Satisfied with your topics? ['y': run model on test data, 'n': choose number of topics again] ")
	
	if topic_satis == 'n':
		continue
	elif topic_satis == 'y':
		apply_model()
		sentiment_analysis()
		rotten_tomato_analysis()
		break
	else:
		break
print()

