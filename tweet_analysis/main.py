from nltk.corpus import twitter_samples
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import re
import string
import random


# Performing normalization with lemmatization and removes noise
def remove_noise(tweet_tokens, stop_words=()):
    cleaned_tokens = []
    for token, tag in pos_tag(tweet_tokens):
        token = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', token)
        token = re.sub("(@[A-Za-z0-9_]+)", "", token)

        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)
        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


def get_all_words(cleaned_tokens_list):  # Calculating word density
    for tokens in cleaned_tokens_list:
        for token in tokens:
            yield token


# Creating a dictionary of cleaned tokens
def get_tweets_for_model(cleaned_tokens_list):
    for tweet_tokens in cleaned_tokens_list:
        yield dict([token, True] for token in tweet_tokens)


if __name__ == "__main__":

    # String varaiables of the dataset
    positive_tweets = twitter_samples.strings('positive_tweets.json')
    negative_tweets = twitter_samples.strings('negative_tweets.json')
    positive_cleaned_tokens_list = []
    negative_cleaned_tokens_list = []
    text = twitter_samples.strings('tweets.20150430-223406.json')
    stop_words = stopwords.words('english')

    # Tokenized variables of dataset
    # tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    positive_tweet_tokens = twitter_samples.tokenized('positive_tweets.json')
    negative_tweet_tokens = twitter_samples.tokenized('negative_tweets.json')

    # Cleaning the noise in the tweet tokens
    for tokens in positive_tweet_tokens:
        positive_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    for tokens in negative_tweet_tokens:
        negative_cleaned_tokens_list.append(remove_noise(tokens, stop_words))

    # Frequency distribution for cleaned words
    all_pos_words = get_all_words(positive_cleaned_tokens_list)
    freq_dist_pos = FreqDist(all_pos_words)

    # Creating positive and negative dictionaries
    positive_tokens_for_model = get_tweets_for_model(
        positive_cleaned_tokens_list)
    negative_tokens_for_model = get_tweets_for_model(
        negative_cleaned_tokens_list)

    # Creating the final dataset for training
    positive_dataset = [(tweet_dict, "Positive")
                        for tweet_dict in positive_tokens_for_model]

    negative_dataset = [(tweet_dict, "Negative")
                        for tweet_dict in negative_tokens_for_model]

    dataset = positive_dataset + negative_dataset

    # Shuffling for suedo-randomness and avoid bias
    random.shuffle(dataset)

    # Dividing shuffled data into train and test data
    train_data = dataset[:7000]
    test_data = dataset[7000:]

    # Initializing the classifier
    classifier = NaiveBayesClassifier.train(train_data)

    custom_tweet = "I ordered just once from TerribleCo, they screwed up, never used the app again."

    custom_tokens = remove_noise(word_tokenize(custom_tweet))

    print(custom_tweet, classifier.classify(
        dict([token, True] for token in custom_tokens)))
