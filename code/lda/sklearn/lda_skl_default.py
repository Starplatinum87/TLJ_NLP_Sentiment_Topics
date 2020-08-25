import pandas as pd
import numpy as np
from pprint import pprint

from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.cluster import KMeans

import time

import pickle

# Import negative reviews & stop words
tlj = pickle.load(open('../../../data/pickles/reviews/imdb_tlj_reviews_sentiments_negative.pkl', 'rb'))
stop_words = pickle.load(open('../../../data/pickles/processing/stop_words.pkl', 'rb'))

# Count vectorizer
vectorizer = CountVectorizer(stop_words=stop_words,token_pattern='[a-zA-Z0-9]{3,}',)

# Use a list of the full documents as the input, not the tokens
data_vectorized=vectorizer.fit_transform(tlj['Reviews'])  

# Build sklearn LDA model
skl_lda_model = LatentDirichletAllocation(n_components=20,        # Let's start on the higher end of topics
                                         max_iter=10, 
                                         learning_method='batch',
                                         random_state=100, 
                                         batch_size=128, 
                                         evaluate_every= -1,  # Don't compute perplexity with every iteration
                                         n_jobs  = -1         # Use all available CPUs
                                         )        

# Fit model
start_time = time.time()
skl_lda_model.fit(data_vectorized)
end_time = time.time()

# Print metrics and params
print("Model Fit Time:", end_time-start_time)
print("Log-Likelihood: ", skl_lda_model.score(data_vectorized))
print("Perplexity: ", skl_lda_model.perplexity(data_vectorized))
pprint(skl_lda_model.get_params)

# Save results
pickle.dump(vectorizer, open('../../../data/pickles/lda/lda_skl_default_vectorizer.pkl', 'wb'))
pickle.dump(data_vectorized, open('../../../data/pickles/lda/lda_skl_default_data_vectorized.pkl', 'wb'))
pickle.dump(skl_lda_model, open('../../../data/pickles/lda/lda_skl_default_model.pkl', 'wb'))