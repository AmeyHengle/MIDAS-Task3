# Custom functions for data prepraration and inference generation

import pandas as pd
import numpy as np
import re
import string
from matplotlib import pyplot as plt
import pickle
from string import punctuation

import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import LabelEncoder



# Reading dataframe
def read_data(path):
  data = pd.read_csv(path)
  data = data.fillna('')
  return data


# Extract all stopwords from a text file and store them in a list.
def get_stopwords(textfile):
  stopwords = []
  file = open(textfile, "r", encoding = "utf-8")
  for line in file:
    line = line.replace("\n","")
    stopwords.append(line)
  file.close()
  return stopwords


# Tokenize input text to integer-id sequences using keras Tokenizer.
def tokenize_text(corpus, x_train, x_val):
  tokenizer = Tokenizer(oov_token = "[OOV]")
  tokenizer.fit_on_texts(corpus)
  x_train_tokenized = tokenizer.texts_to_sequences(x_train)
  x_val_tokenized = tokenizer.texts_to_sequences(x_val)

  return tokenizer, x_train_tokenized, x_val_tokenized


# Padding all text sequences to get uniform-length input. 
def pad_text_sequence(x_tokenized, pad_length, pad_type = "post", truncate_type = "post"):

  x_padded = sequence.pad_sequences(sequences = x_tokenized, 
                                          padding = pad_type,
                                          truncating = truncate_type,
                                          maxlen = pad_length)  
  return x_padded


# Get top-n classes (y) based on frequency
def get_frequent_classes(df, column_name, top_n = 2, min_percent = None):
    
    freq_class_list = []
    if min_percent:
        freq_class_list = (df[column_name].value_counts() * 100 / df[column_name].value_counts().sum())
        freq_class_list = [x for x,y in zip(freq_class_list.keys(), freq_class_list) if y >= min_percent]
    else:
        freq_class_list = df[column_name].value_counts()[:top_n].keys().values.tolist()
   
    print("\nRecords Selected: ", freq_class_list)
    new_df = df[df[column_name].isin(freq_class_list)]
    
    print("\nRecords Removed: ",len(df) - len(new_df))
    return new_df

# Enconde target variabels (class labels) to integers.
def get_label_encoding(labels):
  le = LabelEncoder()
  le.fit(np.unique(labels))
  label_encodings = le.transform(labels)
  
#   print("Mapping:")
  le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
#   print(le_name_mapping)
  label_encodings = label_encodings.reshape(label_encodings.shape[0], -1)
  return label_encodings, le_name_mapping


def prepare_training_data(x_train, x_val, max_seq_len, padding_type, truncating_type):
#     x_train = D_train[text_label].values.tolist()
#     x_val = D_val[text_label].values.tolist()


    corpus = x_train + x_val
    tokenizer, x_train_tokenized, x_val_tokenized = tokenize_text(corpus,x_train, x_val)

    print('\nx_train_tokenized:',len(x_train_tokenized),'\nx_val_tokenized:',len(x_val_tokenized),
          "\nTotal Vocab: ",len(tokenizer.word_counts))

    # Pad Tweets
    x_train_padded = pad_text_sequence(x_train_tokenized, max_seq_len, padding_type, truncating_type)
    x_val_padded = pad_text_sequence(x_val_tokenized, max_seq_len, padding_type, truncating_type)

    print('\nx_train_padded:',x_train_padded.shape,'\nx_val_padded',x_val_padded.shape,"\n")

    return tokenizer, x_train_padded, x_val_padded


def get_word_embeddings(filepath, vocab, ft = False, save_embeddings = False):
  
  binary = False
  embedding_dimension = 0
  embedding_dict = {}

  if ft == True:
    word_vectors = fasttext.load_model(filepath)
    embedding_dimension = len(get_word_vector(list(word_vectors.get_words())[0]))
    print("File loaded. Total Words: {},\t Embedding Dimension: {}".format(len(word_vectors.get_words()), embedding_dimension))

    for word in vocab:
      try:
        wv = word_vectors.get_word_vector[word]
        embedding_dict[word] = wv

      except Exception as e:
        print("Exception reading vector for word:  {}, \n Exception : {} \n".format(word, e))
        continue

    print("Total embeddings found: {}\n\n".format(len(embedding_dict)))

  else:
    if ".bin" in filepath :
      print("Processing binary file")
      binary = True
      
    print("Loading vectors from: {} \n".format(filepath))
    word_vectors =  KeyedVectors.load_word2vec_format(filepath,binary=binary)

    embedding_dimension = len(list(word_vectors.vectors)[0])
    print("File loaded. Total Words: {},\t Embedding Dimension: {}".format(len(word_vectors.vocab), embedding_dimension))
  
  for word in vocab:
    try:
      wv = word_vectors.wv[word]
      embedding_dict[word] = wv

    except Exception as e:
      print("Exception reading vector for word:  {}, \n Exception : {} \n".format(word, e))
      continue

  print("Total embeddings found: {}\n\n".format(len(embedding_dict)))

  if save_embeddings == True:
    output_file = filepath.split('.')[0] + '.pkl'
    with open(output_file, 'wb') as handle:
        pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Word embeddings saved to {} \n\n".format(output_file))

  return embedding_dict


def get_embedding_matrix(vocab, embedding_dict_file = "", embedding_dict = {}, embedding_dimension = 100):
  except_count = 0

  if not embedding_dict:
    try:
        print("Loading embeddings from: ",embedding_dict_file)
        with open(embedding_dict_file, "rb") as f:
            embedding_dict = pickle.load(f)
    except Exception as e:
        print("\nException: ",e)
        
#   vocab_size = len(embedding_dict.keys()) + 1
  vocab_size = len(vocab) + 1
  embedding_matrix = np.zeros((vocab_size, embedding_dimension))


  for i, word in enumerate(vocab):
    try:
      embedding_matrix[i] = embedding_dict[word]
    except Exception as e:
#       print("Exception reading vector for word:  {}, \n Exception : {} \n".format(word, e))
      except_count += 1
      continue

  print("\nTotal words processed: {}".format(len(embedding_matrix) - except_count))
  print("Words not found: ", except_count)
    
  return embedding_matrix



# Different methods of getting sentence/document embeddings from word vectors. 
def get_sentence_embedding(embedding_matrix, corpus, option='bow'):
    all_sentence_embeddings = []
    if option == 'bow':
        for row in corpus:
            sentence_embedding = np.zeros(300)
            for loc, value in list(zip(row.indices, row.data)):
                sentence_embedding = sentence_embedding + value*embedding_matrix[loc]
            if row.data.shape[0] != 0:
                sentence_embedding = sentence_embedding/row.data.shape[0]
            all_sentence_embeddings.append(sentence_embedding)
        all_sentence_embeddings = np.array([np.array(x) for x in all_sentence_embeddings])
        return all_sentence_embeddings
        
    elif option == 'tfidf':
        for row in corpus:
            sentence_embedding = np.zeros(300)
            for loc, value in list(zip(row.indices, row.data)):
                sentence_embedding = sentence_embedding + value*embedding_matrix[loc]
            all_sentence_embeddings.append(sentence_embedding)
        all_sentence_embeddings = np.array([np.array(x) for x in all_sentence_embeddings])
        return all_sentence_embeddings
    
    else:
        print("Invalid option")
        return text


def load_from_pickle(filepath):
  file = open(filepath, "rb")
  data = pickle.load(file)
  return data
    
    
# Save Fasttext trained embeddings to a .vec file.
def save_fasttext_embeddings(model, output_file):
    file = open(output_file, "w",encoding='utf')
    words = model.get_words()
    print('Input Vocab:\t',str(len(words)), "\nModel Dimensions: ",str(model.get_dimension()))
    cnt = 0
    for w in words:
        v = model.get_word_vector(w)
        vstr = ""
        for vi in v:
            vstr += " " + str(vi)
        try:
            row = w + vstr + "\n"
            file.write(row)
            cnt = cnt + 1
        except IOError as e:
            if e.errno == errno.EPIPE:
                pass
    print('Total words processed: ',cnt)
    

# Evaluate model performace timeline.
def plot_results(model_history):

  # Train vs Val accuracy
  plt.title("model accuracy timeline")
  plt.xlabel("epoch")
  plt.ylabel("acc")
  plt.plot(model_history.history['acc'])
  plt.plot(model_history.history['val_acc'])
  plt.legend(["train", "val"])
  plt.show()

  # Train vs Val loss
  plt.title("model loss timeline")
  plt.xlabel("epoch")
  plt.ylabel("loss")
  plt.plot(model_history.history['loss'])
  plt.plot(model_history.history['val_loss'])
  plt.legend(["train", "val"])
  plt.show()


# Save model and model weights to a local file.
def save_model_h5(model, name):
  try:
    model.save(name + ".h5")
    model.save_weights(name + "_weights.h5")
    print("Saved as ",name)
  except Exception as e:
    print(e)

def get_best_model(results, metric = "accuracy"):
    
    max_score = 0
    best_model = {"name" : "", "input_feature": ""}
    for model in results:
        if (model[metric] > max_score):
            best_model["name"] = model["Model"]
            best_model["input_feature"] = model["Input Feature"]
            max_score = model[metric]
    
    return best_model

def save_model_as_pkl(model, filename):
    pickle.dump(model, open(filename + ".pkl", 'wb'))

def load_model_pkl(filename):
    model = pickle.load(open(filename, 'rb'))
    return model
