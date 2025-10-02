

"""
A simple local chatbot using TF-IDF and cosine similarity (scikit-learn).
No API keys required.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import process, fuzz
import re

# Example questions and responses
examples = [
    # Greetings
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings", "yo", "hiya", "howdy",
    # Farewells
    "bye", "goodbye", "see you", "see ya", "later", "farewell", "catch you later", "exit", "quit",
    # Asking about the bot
    "what is your name", "who are you", "tell me about yourself", "what can you do", "who made you", "how do you work", "are you human", "are you a robot", "are you sentient",
    # Well-being
    "how are you", "how's it going", "how do you feel", "what's up", "how are you doing today",
    # Gratitude
    "thank you", "thanks", "thx", "cheers", "much appreciated",
    # Jokes
    "tell me a joke", "make me laugh", "say something funny", "do you know any jokes", "give me a joke",
    # Weather
    "what's the weather", "is it raining", "is it sunny", "how's the weather today", "will it rain tomorrow",
    # Time
    "what time is it", "can you tell me the time", "current time", "do you know the time",
    # Date
    "what's the date", "what day is it", "can you tell me the date", "current date",
    # Personal questions
    "where are you from", "do you have friends", "do you have feelings", "do you like music", "what's your favorite color",
    # Help
    "help", "can you help me", "i need help", "what can you help with", "how do i use you",
    # Small talk
    "what's up", "how's life", "how's your day", "what are you doing", "what's new",
    # Random
    "sing a song", "tell me something interesting", "surprise me", "do you dream", "do you sleep"
]
responses = [
    # Greetings
    "Hello! How can I help you today?", "Hi there!", "Hey! What's up?", "Good morning!", "Good afternoon!", "Good evening!", "Greetings!", "Yo!", "Hiya!", "Howdy!",
    # Farewells
    "Goodbye! Have a great day!", "See you next time!", "Take care!", "Catch you later!", "Farewell!", "Bye!", "See ya!", "Exiting chat. Bye!",
    # Asking about the bot
    "My name is ChatBot. What's yours?", "I'm a Python chatbot here to chat with you!", "I can answer questions, tell jokes, and keep you company.", "I was created by a developer using Python.", "I'm a virtual assistant, not a human.", "I'm not sentient, but I'm here to help!",
    # Well-being
    "I'm just a bot, but I'm doing well! How are you?", "I'm always ready to chat! How are you doing?", "I'm here to help! How's your day?", "I'm doing great, thanks for asking!",
    # Gratitude
    "You're welcome!", "No problem!", "Anytime!", "Glad I could help!", "Cheers!",
    # Jokes
    "Why did the computer show up at work late? It had a hard drive!", "Why was the math book sad? It had too many problems.", "Why don't programmers like nature? It has too many bugs.", "Why did the scarecrow win an award? Because he was outstanding in his field!", "Why do cows have hooves instead of feet? Because they lactose!",
    # Weather
    "I can't check the weather, but I hope it's nice where you are!", "I'm not connected to weather services, but I hope it's sunny!", "I wish I could tell you, but I don't have weather data.", "Sorry, I can't check the weather right now.",
    # Time
    "I can't tell the exact time, but you can check your device clock!", "Sorry, I don't have access to the current time.", "I don't know the time, but it's always a good time to chat!",
    # Date
    "I can't tell the date, but your device can!", "Sorry, I don't have access to the current date.", "I'm not sure of the date, but I'm here for you any day!",
    # Personal questions
    "I'm just a program, so I don't have a hometown.", "I don't have friends, but I love chatting with you!", "I don't have feelings, but I'm here to listen.", "I don't listen to music, but I think it's great!", "I don't have a favorite color, but I like all colors!",
    # Help
    "I'm here to help! Ask me anything.", "I can answer questions, tell jokes, and chat with you.", "Just type your question or message and I'll respond!", "I'm a simple chatbot, so I can help with basic conversation.",
    # Small talk
    "Not much, just chatting with you!", "Life's good in the digital world!", "My day is going well, thanks!", "I'm just here, ready to chat!", "Nothing new, but I'm happy to talk!",
    # Random
    "I can't sing, but I can tell you a joke!", "Here's something interesting: Honey never spoils.", "Surprise! You're awesome!", "I don't dream, but I imagine chatting with you!", "I don't sleep, I'm always here!"
]


# TF-IDF vectorizer setup
vectorizer = TfidfVectorizer().fit(examples)
example_vecs = vectorizer.transform(examples)

# Simple context tracking
conversation_history = []

def normalize(text):
    # Lowercase, remove punctuation, and extra spaces
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def is_math_expression(text):
    # Detect simple math expressions (e.g., 2 + 2, 5*3, what is 7/2)
    text = text.lower().replace('what is', '').replace('calculate', '').replace('?', '').strip()
    # Only allow numbers and math operators
    if re.match(r'^[\d\s\+\-\*/\.]+$', text):
        return True
    return False

def eval_math_expression(text):
    try:
        # Evaluate safely: only numbers and operators
        result = eval(text, {"__builtins__": None}, {})
        return f"The answer is {result}."
    except Exception:
        return "Sorry, I couldn't calculate that."

def get_response(user_input):
    user_input_norm = normalize(user_input)
    # Add user input to context
    conversation_history.append({"role": "user", "content": user_input})

    # Math expression detection
    if is_math_expression(user_input_norm):
        answer = eval_math_expression(user_input_norm)
        conversation_history.append({"role": "bot", "content": answer})
        return answer

    # Fuzzy match to examples
    match, score, idx = process.extractOne(user_input_norm, examples, scorer=fuzz.token_sort_ratio)
    if score > 80:
        reply = responses[idx]
        conversation_history.append({"role": "bot", "content": reply})
        return reply

    # TF-IDF similarity fallback
    user_vec = vectorizer.transform([user_input_norm])
    sims = cosine_similarity(user_vec, example_vecs)[0]
    best_idx = sims.argmax()
    if sims[best_idx] > 0.3:
        reply = responses[best_idx]
        conversation_history.append({"role": "bot", "content": reply})
        return reply

    # Simulate ChatGPT-like open-ended responses
    # Use context to generate a more conversational reply
    if len(conversation_history) > 2:
        last_user = conversation_history[-2]["content"]
        reply = f"Earlier you said: '{last_user}'. Can you tell me more about that?"
    elif "who" in user_input_norm or "what" in user_input_norm or "why" in user_input_norm or "how" in user_input_norm:
        reply = "That's an interesting question! What do you think?"
    elif "feel" in user_input_norm or "think" in user_input_norm:
        reply = "I don't have feelings, but I'm curious about yours!"
    else:
        reply = "I'm here to chat! Ask me anything or tell me more."
    conversation_history.append({"role": "bot", "content": reply})
    return reply

def main():
    print("ChatBot: Hello! Type 'bye' or 'exit' to end the chat.")
    while True:
        user_input = input("You: ")
        if 'bye' in user_input.lower() or 'exit' in user_input.lower():
            print("ChatBot: Goodbye! Have a great day!")
            break
        response = get_response(user_input)
        print(f"ChatBot: {response}")

if __name__ == "__main__":
    main()
