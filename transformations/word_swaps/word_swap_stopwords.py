"""
Word Swap by Stopwords
-------------------------------
"""


import pickle

from textattack.shared import utils

from .word_swap import WordSwap


class WordSwapStopwords(WordSwap):
    """Transforms an input by replacing its words with stopwords."""

    def __init__(self, max_candidates=-1, **kwargs):
        super().__init__(**kwargs)
        self.max_candidates = max_candidates
        self.pos_dict = {"ADJ": "adj", "NOUN": "noun", "ADV": "adv", "VERB": "verb"}

    def _get_transformations(self,current_text,original_text,indices_to_modify):
        # words = current_text.words
        transformed_texts = []
        for i in indices_to_modify:
            replacement_words = ['yourself', 'until', 'such', 'why', 'hers', 'the', 'were', "that'll", 'myself', 'weren', 'once', 'up', "don't", 'each', "haven't", 'himself', 'my', 'an', 'because', 'they', 'on', 'his', 'a', 'and', 'where', 'most', 'over', 'at', 'your', 'wasn', "won't", 'who', 'now', "wasn't", 'with', 'during', "she's", 'there', 'that', 'this', 'ours', 'did', "you'd", 'what', 'nor', 'for', 'of', 'you', 'into', 'ourselves', 'some', 'is', 'only', 'her', 'above', 'be', 'in', "isn't", 'as', 'our', 'itself', 'does', 'when', "needn't", 'if', 'she', 'can', 'him', 'had', 'down', 'by', 'before', 'which', 'between', 'herself', 'have', 'further', 'other', 'both', 'so', 'any', 'to', 'their', 'below', 'them', 'doing', 'very', "you'll", "mightn't", 'through', "you're", 'are', 'those', 'how', 'having', 'yourselves', 'not', "it's", 'me', 'than', 'off', "hasn't", "shouldn't", 'just', 'too', 'but', "couldn't", 'after', 'more', "wouldn't", 'whom', 'themselves', "aren't", 'we', 'again', 'was', 'own', 'same', 'being', "mustn't", 'yours', 'theirs', 'he', 'should', 'has', 'do', 'no', 'or', 'am', 'it', 'been', "you've", 'while', 'its', 'against', 'from', 'under', 'all', "hadn't", "weren't", 'here', "didn't", 'then', 'these', "doesn't", 'few', 'about', 'will', 'out']
            for r in replacement_words:
                transformed_texts.append(current_text.replace_word_at_index(i, r))
        return transformed_texts


    def extra_repr_keys(self):
        return ["max_candidates"]
    

def recover_word_case(word, reference_word):
    """Makes the case of `word` like the case of `reference_word`.

    Supports lowercase, UPPERCASE, and Capitalized.
    """
    if reference_word.islower():
        return word.lower()
    elif reference_word.isupper() and len(reference_word) > 1:
        return word.upper()
    elif reference_word[0].isupper() and reference_word[1:].islower():
        return word.capitalize()
    else:
        # if other, just do not alter the word's case
        return word
