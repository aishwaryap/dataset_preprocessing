#!/usr/bin/python

import os
import re
import nltk
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


def normalize_string(string):
    stemmer = nltk.stem.snowball.SnowballStemmer("english")
    string = string.lower().strip()
    string = re.sub('[^a-z]', ' ', string)      # Replace anything other than letters with space
    string = re.sub('\s+', ' ', string)         # Replace a sequence of spaces with a single space
    tokens = string.split()
    stemmed_tokens = [str(stemmer.stem(token)) for token in tokens]
    stopwords = set(nltk.corpus.stopwords.words('english'))
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
