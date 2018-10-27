# OffComBr3 loader
# Total dataset has 1034 sentences
# Train: 700 sentences + 231 validation sentences
# Test: 103 sentences
#
# OffComBR2
# Total dataset has 1251 sentences
# Train: 846 sentences + 280 validation sentences
# Test: 125 sentences

from __future__ import print_function
import numpy as np
import nltk
import nltk.corpus as corpus
import nltk.stem.snowball as snowball
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import re
import random

ptbr_stem = snowball.SnowballStemmer('portuguese').stem

def create_lexicon(dataset):
    global max_doc_length
    lexicon = []
    with open(dataset, 'r') as f:
        contents = f.readlines()[7:]
        for line in contents:
            document = line.split(',')[-1]
            document = re.sub("'", '', document, flags = re.M)
            document = word_tokenize(document)

            for word in document:
                if word.lower() not in corpus.stopwords.words('portuguese'):
                    lexicon.append(word.lower())

    lexicon_lemmatized = []

    for token in lexicon:
        lexicon_lemmatized.append(ptbr_stem(token))

    w_counts = Counter(lexicon)
    l2 = []
    for w in w_counts:
        if w_counts[w] > 3:
            l2.append(w)

    return l2 # l2 has 478 tokens, for OffComBR3 and 551 for OffComBR2

# BOW approach with simple count
def sample_handling(dataset, lexicon):
    feature_set = []
    with open(dataset, 'r') as f:
        contents = f.readlines()[7:]
        for line in contents:
            document = line.split(',')[-1]
            document = re.sub("'", '', document, flags = re.M)

            # Tokenize
            document = word_tokenize(document)

            # Remove stopwords
            document = [word for word in document
                if word.lower() not in corpus.stopwords.words('portuguese')]

            # Stem
            document = [ptbr_stem(word) for word in document]
            features = np.zeros(len(lexicon))

            for token in document:
                if token.lower() in lexicon:
                    index = lexicon.index(token.lower())
                    label = line.split(',')[0]
                    features[index] += 1 # simple count


            features = list(features)

            if line.split(',')[0] == 'yes':
                feature_set.append([np.array(features), [1,0]])
            else:
                feature_set.append([np.array(features), [0,1]])
    return feature_set, None

def get_embed(filepath, embedding_index):
    with open(filepath, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs
    return embedding_index

def embedding(dataset, embed_method, lexicon):
    embedding_index, dim = dict(), 0
    features_set = []
    labels = []
    t = Tokenizer()
    docs = []
    with open(dataset, 'r') as f:
        contents = f.readlines()[7:]
        for line in contents:
            document, label = line.split(',')[-1], line.split(',')[0]
            document = re.sub("'", '', document, flags = re.M)
            docs.append(re.sub('\r\n', '', document, flags = re.M))
            if label == 'yes':
                labels.append([1, 0])
            else:
                labels.append([0, 1])

        t.fit_on_texts(docs)
        vocab_size = len(t.word_index) + 1
        encoded_docs = t.texts_to_sequences(docs) # hash encoding

        sequence_length = len(lexicon)
        for doc in encoded_docs:
            length = 0
            for token in doc:
                length += 1
            if length > sequence_length:
                sequence_length = length

        padded_docs = pad_sequences(encoded_docs, maxlen=sequence_length, padding='post')

        if embed_method == 'glove50':
            embedding_index = get_embed('embeddings/glove_s50.txt', embedding_index)
            dim = 50

        elif embed_method == 'glove100':
            embedding_index = get_embed('embeddings/glove_s100.txt', embedding_index)
            dim = 100
 
        elif embed_method == 'glove300':
            embedding_index = get_embed('embeddings/glove_s300.txt', embedding_index)
            dim = 300
           
        elif embed_method == 'wang2vec50':
            embedding_index = get_embed('embeddings/skip_s50.txt', embedding_index)
            dim = 50

        elif embed_method == 'wang2vec100':
            embedding_index = get_embed('embeddings/skip_s100.txt', embedding_index)
            dim = 100

        elif embed_method == 'wang2vec300':
            embedding_index = get_embed('embeddings/skip_s300.txt', embedding_index)
            dim = 300

        
        embedding_matrix = np.zeros((vocab_size, dim))

        for word, i in t.word_index.items():
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        for index, padded_doc in enumerate(padded_docs):
            features_set.append([padded_doc, labels[index]])

    return features_set, embedding_matrix

def create_feature_sets_and_labels(dataset, embed_method, test_size=0.1):
    lexicon = create_lexicon(dataset)
    features = []

    # This is for selecting methods for handling the dataset
    if embed_method == 'none':
        features, embed = sample_handling(dataset, lexicon)
    else:
        # features = hash_encoding(dataset, lexicon)
        # This is for selecting which embedding to use
        features, embed = embedding(dataset, embed_method, lexicon)

    features = np.array(features)

    random.shuffle(features)

    testing_size = int(test_size * len(features))

    train_x = list(features[:, 0][:-testing_size])
    train_y = list(features[:, 1][:-testing_size])
    test_x = list(features[:, 0][-testing_size:])
    test_y = list(features[:, 1][-testing_size:])

    return train_x, train_y, test_x, test_y, lexicon, embed
