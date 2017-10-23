#!/usr/bin/env python
# -*- coding: UTF-8 -*-
__author__ = "Josh Montague"
__license__ = "MIT License"

import argparse
import fileinput
try:
    import ujson as json
except ImportError:
    import json
import glob
import logging
import multiprocessing as mp 
import os
import re
import sys

from gensim.models.keyedvectors import KeyedVectors 
from gensim.models import Word2Vec
import numpy as np
from nltk.tokenize.casual import TweetTokenizer 
from tweet_parser.tweet import Tweet
from tweet_parser.tweet_parser_errors import NotATweetError


####
# 
def whitespace_tokenizer(s, **kwargs):
    """Tokenize on whitespace"""
    token_list = s.split(' ')
    return token_list 


def nltk_tweet_tokenizer(s, **tokenizer_kwargs): 
    """NTLK TweetTokenizer"""
    kwargs = dict(strip_handles=False, reduce_len=True)
    kwargs.update(**tokenizer_kwargs)
    tokenizer = TweetTokenizer(**kwargs)
    token_list = tokenizer.tokenize(s)
    return token_list 
# 
####


def get_summary_vector(model, token_list):
    """Calculate a dense summary vector from an iterable of individual token vectors.""" 
    n = 0.0 
    summary_vector = np.zeros(model.vector_size, dtype='float32')
    for token in token_list:
        try:
            summary_vector = np.add(summary_vector, model[token]) 
            n += 1.0
        except KeyError: 
            logging.debug('token not in vocab: {}'.format(token))
            continue
    try:
        logging.debug('frac of tokens used in summary vector: {:.3f}'.format(n / len(token_list)))
    except ZeroDivisionError:
        pass
    summary_vector = np.divide(summary_vector, n) if n > 0 else summary_vector
    return summary_vector


def extract_tokens(tweet, tokenizer=nltk_tweet_tokenizer, **tokenizer_kwargs):
    """Apply specified tokenizer to specified Tweet field""" 
    text = tweet.text
    tokens = tokenizer(text, **tokenizer_kwargs)
    return tokens


def parse_tweet(json_string):
    """Parse JSON string to Tweet object. Returns None if parsing fails.""" 
    tweet = None
    try:
        tweet_dict = json.loads(json_string)
        tweet = Tweet(tweet_dict)
    except (json.JSONDecodeError, NotATweetError):
        logging.debug('record is not a Tweet: {}'.format(json_string[:75])) 
    return tweet


def load_model(filepath, keyed_vec=False):
    """
    Instantiate a pre-trained model located at `filepath`. If read-only model vectors 
    were trained by another application, set `keyed_vec=True`. Otherwise, word2vec model 
    is assumed. 
    """
    if keyed_vec:
        model = KeyedVectors.load(filepath)
    else:
        model = Word2Vec.load(filepath)
    return model


def load_GNews_model():
    """
    Convenience function for loading the pre-trained Google News word2vec model vectors 
    published with the original work. For more information see: 
    https://code.google.com/archive/p/word2vec/ 
    """
    model = KeyedVectors.load_word2vec_format('rdata/GoogleNews-vectors-negative300.bin', binary=True) 
    return model


def create_model(datapath, **w2vkw):
    """Generate an embedding model from the ndjson at filepath"""
    doc_generator = TweetSentences(datapath, tokenizer=nltk_tweet_tokenizer)
    kwargs = dict(size=300, 
                  window=5,
                  sg=0,
                  min_count=20,
                  workers=mp.cpu_count())
    kwargs.update(**w2vkw)
    model = Word2Vec(doc_generator, **kwargs)
    return model


def parser_setup():
    """Shield CLI pieces from imports"""
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="increase output verbosity")
    parser.add_argument("-i", "--infile", action="store", type=str, default=sys.stdin, 
                        help="specify input data files")
    args = parser.parse_args()
    # use a simple logger - get the level from the cmd line
    loglevel = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', 
                        stream=sys.stderr, level=loglevel)
    logging.debug('logging enabled - preparing for work') 
    return args


def strip_urls(text, replace=' '):
    """A simple preprocessing function that strips URLs from text, based on a regex."""
    # simple python regex for URLs: bit.ly/PyURLre
    url_pattern = '(https?://)?(\\w*[.]\\w+)+([/?=&]+\\w+)*'
    url_re = re.compile(url_pattern)
    stripped = re.sub(url_re, replace, text)
    return stripped


class TweetSentences:
    """
    Class that allows for lazy iteration over input data (newline-delimited JSON, or files of the 
    same), and specific tokenization of Tweet text.
    """
    def __init__(self, input_path, tokenizer=nltk_tweet_tokenizer):
        self.tokenizer = tokenizer
        self.filenames = [] 
        if os.path.isfile(input_path):
            self.filenames.append(input_path)
        else:
            for expanded_path in glob.glob(input_path): 
                for path, _, files in os.walk(expanded_path):
                    if files != []:
                        self.filenames.extend(
                            [path.rstrip('/') + '/' + file for file in files])
        logging.info('reading {} input files from: {}'.format(len(self.filenames), input_path))
        logging.info('using tokenizer: {}'.format(tokenizer))

    def __iter__(self):
        """Define the behavior of our sentence iterator"""
        with fileinput.input(files=self.filenames, openhook=fileinput.hook_compressed) as instream:
            for line in instream:
                if fileinput.filename().endswith('.gz'):
                    # pretty sure fileinput is supposed to handle this for me ¯\_(ツ)_/¯ 
                    line = line.decode()
                tw = parse_tweet(line)
                tokens = extract_tokens(tw, tokenizer=self.tokenizer) if tw else []
                yield tokens


# this is only applicable to Phase 1 (model application to stream of new tweets)
if __name__ == '__main__':
    args = parser_setup()
    model = load_GNews_model()
    for line in args.infile: 
        tw = parse_tweet(line)
        if tw:
            tokens = extract_tokens(tw)
            summary_vector = get_summary_vector(model, tokens) 
            print('tokens: {}'.format(tokens))
            print('summary vector[:10]: {}'.format(summary_vector[:10]))
            print('\n')
