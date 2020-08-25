from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV

import time

import pickle

# Import negative reviews & stop words
tlj = pickle.load(open('../../../data/pickles/reviews/imdb_tlj_reviews_sentiments_negative.pkl', 'rb'))
stop_words = pickle.load(open('../../../data/pickles/processing/stop_words.pkl', 'rb'))

# Count vectorizer
vectorizer = CountVectorizer(stop_words=stop_words,token_pattern='[a-zA-Z0-9]{3,}',)

# Use a list of the full documents as the input, not the tokens
data_vectorized=vectorizer.fit_transform(list(tlj['Reviews']))  

# Define Search Params
search_params = {'n_components': [5, 10, 15, 20, 25, 30, 35, 40], 'learning_decay': [.5, .7, .9]} # Search broadly, then narrow

# Init the Model
lda = LatentDirichletAllocation(n_jobs=-1, learning_method='batch')

# Init GridSearch instance
model = GridSearchCV(lda, param_grid=search_params, cv=5)

# Run Grid Search
start_time = time.time()
model.fit(data_vectorized)
end_time = time.time()

# Best Model
skl_best_lda_model = model.best_estimator_

# Save data
pickle.dump(model, open('../../../data/pickles/lda/lda_skl_grid_model_1.pkl', 'wb'))
pickle.dump(data_vectorized, open('../../../data/pickles/lda/lda_skl_grid_data_vectorized_1.pkl', 'wb'))
pickle.dump(vectorizer, open('../../../data/pickles/lda/lda_skl_vectorizer_1.pkl', 'wb'))
pickle.dump(skl_best_lda_model, open('../../../data/pickles/lda/lda_skl_grid_best_model_1.pkl', 'wb'))

# Print stats
print("Best Model's Params: ", model.best_params_)
print("Best Log Likelihood: ", model.best_score_)
print("Model Perplexity: ", skl_best_lda_model.perplexity(data_vectorized))
print("Model Execution Time:", end_time-start_time)
