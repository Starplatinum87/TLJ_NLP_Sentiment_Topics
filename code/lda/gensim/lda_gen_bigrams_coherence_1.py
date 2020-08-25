import pandas as pd
import numpy as np
from pprint import pprint

from nltk.corpus import stopwords

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import matplotlib.pyplot as plt

import pickle

import time

# Import dictionary, corpus and n-grams
corpus = pickle.load(open('../../../data/pickles/lda/lda_gen_bigrams_corpus.pkl', 'rb'))
id2word = pickle.load(open('../../../data/pickles/lda/lda_gen_bigrams_id2word.pkl', 'rb'))
ngram_model = pickle.load(open('../../../data/pickles/lda/lda_gen_bigrams_lemmatized.pkl', 'rb'))


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=num_topics,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

# Number of topics
start = 2
limit = 32
step = 3

start_time = time.time()
# Create model list and associated coherence values
model_list, coherence_values = compute_coherence_values(dictionary=id2word, 
                                                        corpus=corpus, 
                                                        texts=ngram_model, 
                                                        start=start, limit=limit, step=step)
end_time = time.time()                                                       

# Find optimal model by highest coherence value
optimal_model = model_list[coherence_values.index(max(coherence_values))]

# Print topics of optimial model
pprint(optimal_model.print_topics(num_topics=-1)) 

# Print stats
x = range(start, limit, step)
for m, cv in zip(x, coherence_values):
    print("Num Topics =",m, " has a Coherence of", round(cv,4))
print("Coherence Search Execution Time:", str(end_time-start_time))
print("Top Coherence Score:", max(coherence_values))
print("Optimal Topic Number:", str(coherence_values.index(max(coherence_values)) + start))


# Pickle data
pickle.dump(model_list, open('../../../data/pickles/lda/lda_gen_coherence_model_list_1.pkl', 'wb'))
pickle.dump(coherence_values, open('../../../data/pickles/lda/lda_gen_coherence_model_values_1.pkl', 'wb'))
pickle.dump(optimal_model, open('../../../data/pickles/lda/lda_gen_coherence_optimal_model_1.pkl', 'wb'))
pickle.dump((start, limit, step), open('../../../data/pickles/lda/lda_gen_coherence_topic_search_values_1.pkl', 'wb'))
    
# Show graph of coherence values for each n-gram
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()
plt.savefig('../data/images/lda_gen_bigrams_coherence_1.png')
