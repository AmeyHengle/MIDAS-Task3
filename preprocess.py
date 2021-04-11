import pkg_resources
from symspellpy import SymSpell, Verbosity

sym_spell = SymSpell(max_dictionary_edit_distance=3, prefix_length=7)
dictionary_path = pkg_resources.resource_filename(
    "symspellpy", "frequency_dictionary_en_82_765.txt")
if sym_spell.word_count:
    pass
else:
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    

import re
from stop_words import get_stop_words
from language_detector import detect_language


import nltk
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
        
    
class Preprocessing:
    
    def __init__(self):
        
        self.p_stemmer = PorterStemmer()
        self.en_stop = get_stop_words('en')
    
    
    """ ----------------------------- Word Level Preprocessing methods -----------------------------
    """

    def remove_punct(self, w_list):
        """
        Function for filtering out punctuations and numbers.
            @param w_list (list): word list to be processed.
            @return w_list (list): word list with punctuation and numbers filtered out. 
        """   
        return [word for word in w_list if word.isalpha()]


    # selecting nouns
    def get_nouns(self, w_list):
        """
        Function for selecting nouns.
            @param w_list (list): word list to be processed.
            @return w_list (list): word list with only nouns. 
        """ 
        return [word for (word, pos) in nltk.pos_tag(w_list) if pos[:2] == 'NN']


    def get_stemming(self, w_list):
        """
        Function for stemming.
            @param w_list (list): word list to be processed.
            @return w_list (list): word list with stemming. 
        """ 
        # stemming if doing word-wise
        return [self.p_stemmer.stem(word) for word in w_list]


    def remove_stopw(self, w_list):
        """
        Function for removing stopwords.
            @param w_list (list): word list to be processed.
            @return w_list (list): word list without stopwords. 
        """ 
        return [word for word in w_list if word not in self.en_stop]
    

    # typo correction
    def correct_typos(self, w_list):
        """
        Function for correcting typos.
            @param w_list (list): word list to be processed.
            @return w_list (list): word list with typos fixed using symspell. All words with no match up are dropped. 
        """ 
        w_list_fixed = []
        for word in w_list:
            suggestions = sym_spell.lookup(word, Verbosity.CLOSEST, max_edit_distance=3)
            if suggestions:
                w_list_fixed.append(suggestions[0].term)
            else:
                pass
        return w_list_fixed

    
    """ ----------------------------- Sentence Level Preprocessing methods -----------------------------
    """ 

    def clean_text(self, text):
        """
        Function for cleaning a string.
            @param text (string): text to be cleaned.
            @return text (string): cleaned string.  
        """ 
        # normalization 1: xxxThis is a --> xxx. This is a (missing delimiter)
        text = re.sub(r'([a-z])([A-Z])', r'\1\. \2', text)  # before lower case
        # normalization 2: lower case
        text = text.lower()
        # normalization 3: "&gt", "&lt"
        text = re.sub(r'&gt|&lt', ' ', text)
        # normalization 4: letter repetition (if more than 2)
        text = re.sub(r'([a-z])\1{2,}', r'\1', text)
        # normalization 5: non-word repetition (if more than 1)
        text = re.sub(r'([\W+])\1{1,}', r'\1', text)
        # normalization 6: string * as delimiter
        text = re.sub(r'\*|\W\*|\*\W', '. ', text)
        # normalization 7: stuff in parenthesis, assumed to be less informal
        text = re.sub(r'\(.*?\)', '. ', text)
        # normalization 8: xxx[?!]. -- > xxx.
        text = re.sub(r'\W+?\.', '.', text)
        # normalization 9: [.?!] --> [.?!] xxx
        text = re.sub(r'(\.|\?|!)(\w)', r'\1 \2', text)
        # normalization 10: ' ing ', noise text
        text = re.sub(r' ing ', ' ', text)
        # normalization 12: phrase repetition
        text = re.sub(r'(.{2,}?)\1{1,}', r'\1', text)

        return text.strip()


    # language detection
    def check_lang(self, text, lang_list = {'English', 'French'}):
        """
        Function for language detection.
            @param text (string): text to be processed.
            @param lang_list(set): allowed languages for the text (Default: English + French)
            @return boolean: (True if text belongs to any one of the languages in lang_list) 
        """ 
        return detect_language(s) in lang_list


    
    def normalize_text(self, text, remove_punct = False, fix_typos = True, remove_stopwords = False, 
                        stemming = False, nouns_only = False):
        """
        Function to get word level preprocessed data from preprocessed sentences.
            @param text (str): sentence to be processed.
            @return w_list (list): word list normalized.
        """ 
        if not text:
            return None
        w_list = word_tokenize(text)
        if remove_punct: w_list = self.remove_punct(w_list)
        if nouns_only: w_list = self.get_nouns(w_list)
        if fix_typos: w_list = self.correct_typos(w_list)            
        if stemming: w_list = self.get_stemming(w_list)
        if remove_stopwords: w_list = self.remove_stopw(w_list)
            
        return " ".join([x for x in w_list])