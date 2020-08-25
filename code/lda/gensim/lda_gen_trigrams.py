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
tlj = pickle.load(open('../../../data/pickles/reviews/imdb_tlj_reviews_sentiments_negative.pkl', 'rb'))
stop_words = pickle.load(open('../../../data/pickles/processing/stop_words.pkl', 'rb'))

# Create a list of tokenized negative reviews
tlj_data = list(tlj['Tokens'])

# Build bigrams model
# Increased threshold from default of 10 to 100 for more restrictive membership in the model
bigram = gensim.models.Phrases(tlj_data, min_count=5, threshold=100) 
trigram = gensim.models.Phrases(bigram[tlj_data], threshold=100)

# Faster way to get a sentence clubbed as a trigram/bigram
trigram_mod = gensim.models.phrases.Phraser(trigram)

# Remove stop words from corpus
tlj_data_nostops = [[word for word in doc if word not in stop_words] for doc in tlj_data]

# Create Bigrams
tlj_data_trigrams = [trigram_mod[doc] for doc in tlj_data_nostops]

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
tlj_data_trigram_lemmatized = lemmatization(tlj_data_trigrams, nlp, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

# Create Dictionary with bigrams
id2word = corpora.Dictionary(tlj_data_trigram_lemmatized)

# Create Corpus
texts = tlj_data_trigram_lemmatized

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
                                     texts=tlj_data_trigram_lemmatized, 
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
print("Trigrams LDA model execution time:", end - start)

# Pickles
pickle.dump(tlj_data_trigram_lemmatized, open('../../../data/pickles/lda/lda_gen_trigrams_lemmatized.pkl', 'wb'))
pickle.dump(id2word, open('../../../data/pickles/lda/lda_gen_trigrams_id2word.pkl', 'wb'))
pickle.dump(corpus, open('../../../data/pickles/lda/lda_gen_trigrams_corpus.pkl', 'wb'))
pickle.dump(lda_model, open('../../../data/pickles/lda/lda_gen_trigrams_model.pkl', 'wb'))