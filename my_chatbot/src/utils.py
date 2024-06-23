import json
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

def load_intents(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        intents = json.load(file)
    return intents

def tokenize_and_lemmatize(sentence):
    tokens = nltk.word_tokenize(sentence)
    lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
    return lemmatized_tokens

def preprocess_training_data(intents):
    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!', '.', ',']
    
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            word_list = tokenize_and_lemmatize(pattern)
            words.extend(word_list)
            documents.append((word_list, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
    
    words = [w for w in words if w not in ignore_words]
    words = sorted(set(words))
    classes = sorted(set(classes))
    
    training = []
    output_empty = [0] * len(classes)
    
    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)
        
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        
        training.append([bag, output_row])
    
    training = np.array(training, dtype=object)
    train_x = np.array(list(training[:, 0]))
    train_y = np.array(list(training[:, 1]))
    
    return words, classes, train_x, train_y

def create_bag_of_words(sentence, words):
    bag = [0] * len(words)
    sentence_words = tokenize_and_lemmatize(sentence)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def save_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
