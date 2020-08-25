import pandas as pd

from nltk.corpus import stopwords

import pickle

# Import tlj df pickle and get negative reviews
tlj_all = pickle.load(open('../data/pickles/reviews/imdb_tlj_reviews_sentiments.pkl', 'rb'))
tlj = tlj_all.loc[tlj_all['google_sentiment']=='negative']
tlj_negative_review_list = list(tlj['Reviews'])

# Create NLTK stopwords combining basic NLTK stop words and custom stop words
more_stop_words = ['film', 'scene', 'much', 'seem', 'otherwise', 'picture', \
                  'star', 'wars', 'starwars', 'last', 'would', 'make', 'well',\
                  'utter', 'proceed', 'video', 'eager', 'necessarily', 'grade',\
                  'elsewhere', 'think', 'movie', 'really']
stop_words = stopwords.words('english') + more_stop_words

more_stop_words_2 = ['like', 'character', 'one', 'jedi', 'characters', 'even', 'could', \
                    'many', 'made', 'also', 'get', 'one', 'movie', 'movies', 'time', 'watch',\
                    'say', 'never', 'back', 'nothing', 'still', 'something', 'come']

stop_words_2 = stop_words + more_stop_words_2


pickle.dump(stop_words, open('../data/pickles/processing/stop_words.pkl', 'wb'))
pickle.dump(stop_words_2, open('../data/pickles/processing/stop_words_2.pkl', 'wb'))
pickle.dump(tlj, open('../data/pickles/reviews/imdb_tlj_reviews_sentiments_negative.pkl', 'wb'))
pickle.dump(tlj_negative_review_list, open('../data/pickles/reviews/imdb_tlj_reviews_list_negative.pkl', 'wb'))
