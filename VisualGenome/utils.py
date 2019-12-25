#!/usr/bin/python

import os
import re
import nltk
import time
from nltk.corpus import wordnet as wn

__author__ = 'aishwarya'


# This dataset has 2 image folders. This is a util to search both and find the image
# verify=True does a more robust check that the image actually exists
def get_image_path(dataset_dir, image_id, verify=False):
    path1 = os.path.join(*[dataset_dir, 'VG_100K', str(image_id) + '.jpg'])
    path2 = os.path.join(*[dataset_dir, 'VG_100K_2', str(image_id) + '.jpg'])
    if os.path.isfile(path1):
        return path1
    else:
        if verify:
            if not os.path.isfile(path2):
                return None
        return path2


def in_vocab(token, word_vectors):
    try:
        _ = word_vectors.get_vector(token)
    except KeyError:
        return False
    return True


def normalize_string(string, word_vectors=None):
    string = string.lower().strip()
    string = re.sub('[^a-z]', ' ', string)      # Replace anything other than letters with space
    string = re.sub('\s+', ' ', string)         # Replace a sequence of spaces with a single space
    tokens = string.split()

    stemmer = nltk.stem.snowball.SnowballStemmer("english")
    stopwords = set(nltk.corpus.stopwords.words('english'))

    if word_vectors is None:
        stemmed_tokens = [str(stemmer.stem(token)) for token in tokens]
    else:
        stemmed_tokens = list()
        for token in tokens:
            if in_vocab(token, word_vectors):
                stemmed_tokens.append(token)
            else:
                stemmed_token = str(stemmer.stem(token))
                if in_vocab(stemmed_token, word_vectors):
                    stemmed_tokens.append(stemmed_token)

    useful_tokens = [token for token in stemmed_tokens if token not in stopwords]
    string = '_'.join(useful_tokens)
    return string.strip()


def get_synset(synset_name):
    try:
        synset = wn.synset(synset_name)
        return synset
    except KeyboardInterrupt:
        raise
    except SystemExit:
        raise
    except:
        return None


def create_dir(path):
    sub_path = os.path.dirname(path)
    if not os.path.exists(sub_path):
        create_dir(sub_path)

    delay = 0.1
    num_delays = 0
    while True:
        try:
            if not os.path.exists(path):
                os.mkdir(path)
            break
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            # Time delay with exponential backoff
            time.sleep(delay)
            delay *= 2
            num_delays += 1
            if num_delays > 10:
                raise RuntimeError('Directory creation failed with race conditions 10 times')


