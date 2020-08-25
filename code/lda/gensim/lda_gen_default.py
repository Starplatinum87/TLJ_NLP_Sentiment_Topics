import pandas as pd 
from pprint import pprint

import pickle

from nltk.corpus import stopwords

import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel

import spacy

import time

# Import negative reviews & stop words
tlj = pickle.load(open('../../../data/pickles/reviews/imdb_tlj_reviews_sentiments.pkl', 'rb'))
stop_words = pickle.load(open('../../../data/pickles/processing/stop_words.pkl', 'rb'))

# Create a list of tokenized negative reviews
tlj_data = list(tlj['Tokens'])

# Remove stop words from corpus
tlj_data_nostops = [[word for word in doc if word not in stop_words] for doc in tlj_data]

# Initialize spaCy 'en' model, keeping only tagger component
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Define lemmatization function
def lemmatization(texts, model, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Lemmatize with bigrams, keeping only noun, adj, vb, adv
tlj_data_lemmatized = lemmatization(tlj_data_nostops, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Create Dictionary with bigrams
id2word = corpora.Dictionary(tlj_data_lemmatized)

# Create Corpus
texts = tlj_data_lemmatized

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Time execution
start = time.time()

# Build LDA model. Using some default parameters here. Will probably tweak.
lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                            id2word=id2word,
                                            num_topics=20,
                                            random_state=100,
                                            update_every=1,
                                            chunksize=100,
                                            passes=10,
                                            alpha='auto',
                                            per_word_topics=True)

end = time.time()

# Display coherence score
coherence_model_lda = CoherenceModel(model=lda_model, 
                                     texts=tlj_data_lemmatized, 
                                     dictionary=id2word, 
                                     coherence='c_v')

# Print topics and keywords
pprint(lda_model.print_topics(num_topics=-1))

# Print perplexity score
print('Perplexity: ', lda_model.log_perplexity(corpus))

# Print coherence score
coherence_lda = coherence_model_lda.get_coherence()
print('Coherence: ', coherence_lda)

# Print execution time
print("Default LDA model execution time:", end - start)

# Pickles
pickle.dump(tlj_data_lemmatized, open('../../../data/pickles/lda/lda_gen_default_lemmatized.pkl', 'wb'))
pickle.dump(id2word, open('../../../data/pickles/lda/lda_gen_default_id2word.pkl', 'wb'))
pickle.dump(corpus, open('../../../data/pickles/lda/lda_gen_default_corpus.pkl', 'wb'))
pickle.dump(lda_model, open('../../../data/pickles/lda/lda_gen_default_model.pkl', 'wb'))