import json
import numpy as np
import random
import nltk
import pickle
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

class Chatbot:
    def __init__(self):
        self.model = load_model('../models/chatbot_model.keras')
        self.words = self.load_pickle('../models/words.pkl')
        self.classes = self.load_pickle('../models/classes.pkl')
        self.intents = self.load_intents('../data/intents.json')

    def load_intents(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            intents = json.load(file)
        return intents

    def load_pickle(self, file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def tokenize_and_lemmatize(self, sentence):
        tokens = nltk.word_tokenize(sentence)
        lemmatized_tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens]
        return lemmatized_tokens

    def bow(self, sentence):
        sentence_words = self.tokenize_and_lemmatize(sentence)
        bag = [0] * len(self.words)
        for s in sentence_words:
            for i, w in enumerate(self.words):
                if w == s:
                    bag[i] = 1
        return np.array(bag)

    def predict_class(self, sentence):
        bow = self.bow(sentence)
        print(f"BOW shape: {bow.shape}, Model input shape: {self.model.input_shape}")
        res = self.model.predict(np.array([bow]))[0]
        ERROR_THRESHOLD = 0.25
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def get_response(self, intents_list, intents_json):
        tag = intents_list[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                return random.choice(i['responses'])

    def chat(self, text):
        ints = self.predict_class(text)
        res = self.get_response(ints, self.intents)
        return res

if __name__ == "__main__":
    bot = Chatbot()
    print("Chatbot is ready to chat! Type 'exit' to end the conversation.")
    while True:
        message = input("You: ")
        if message.lower() == "exit":
            break
        response = bot.chat(message)
        print(f"Bot: {response}")
