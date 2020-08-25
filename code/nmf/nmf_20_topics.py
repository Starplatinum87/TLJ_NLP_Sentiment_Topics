import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.decomposition import NMF
import pickle

from pprint import pprint

import time

# Number of topics
n_components = 20

# Import negative reviews & stop words
tlj_negative_review_list = pickle.load(open('../../data/pickles/reviews/imdb_tlj_reviews_sentiments_negative.pkl', 'rb'))
stop_words_2 = pickle.load(open('../../data/pickles/processing/stop_words_2.pkl', 'rb'))

# Initialize spaCy 'en' model, keeping only tagger component
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Preprocess with TF-IDF
tfidf = TfidfVectorizer(stop_words=stop_words_2)
tlj_tfidf = tfidf.fit_transform(tlj_negative_review_list)

# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return[[word for word in doc if word not in stop_words_2] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent))
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

# Function to display topics
def display_topics(model, feature_names, no_top_words, topic_names=None):
    topic_list = []
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            topics = "Topic: " + str(ix)
        else:
            topics = "Topic:'" + str(topic_names[ix])
        terms = ", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]])
        topic_list.append((topics, terms))
    return topic_list

nmf_model = NMF(n_components=n_components, random_state=42)

start = time.time()
doc_topic = nmf_model.fit_transform(tlj_tfidf)
end = time.time()

nmf_topics = display_topics(nmf_model, tfidf.get_feature_names(), 20)
pprint(nmf_topics)
print("Model Execution Time:", end-start)

pickle.dump(nmf_topics, open('../../data/pickles/nmf/nmf_topics_'+str(n_components)+'.pkl', 'wb'))
pickle.dump(nmf_model, open('../../data/pickles/nmf/nmf_model_'+str(n_components)+'.pkl', 'wb'))