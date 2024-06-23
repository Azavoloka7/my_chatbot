from chatbot import Chatbot

if __name__ == "__main__":
    bot = Chatbot()
    print("Start talking with the bot (type 'quit' to stop)!")
    while True:
        message = input("You: ")
        if message.lower() == "quit":
            print("Goodbye!")
            break
        response = bot.chat(message)
        print("Bot:", response)
