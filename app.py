import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import speech_recognition as sr
import pyttsx3
import base64
import threading

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(page_title="BERTConnect", page_icon="🤖")

# =========================================================
# BACKGROUND IMAGE + MINIMAL STYLE
# =========================================================
def set_background(image_path):
    """Set a background image and unified CSS."""
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    css = f"""
    <style>
    /* Background */
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        font-family: 'Poppins', sans-serif;
    }}

    /* Title */
    h1 {{
        text-align: center;
        color: #111111; /* Dark blackish color */
        font-size: 2.5rem;
        font-weight: 800;
        text-shadow: 1px 1px 3px rgba(255,255,255,0.3);
    }}

    /* Subtitle */
    p {{
        text-align: center;
        color: #e0e0e0;
        font-size: 1.1rem;
        margin-bottom: 20px;
    }}

    /* Input Box */
    div.stTextInput > div > input {{
        background-color: rgba(0, 0, 0, 0.6);
        color: #ffffff;
        border-radius: 10px;
        border: 1px solid #00ffff;
        font-size: 1rem;
        padding: 8px;
    }}

    /* Buttons */
    button[kind="primary"], button[kind="secondary"], div[data-testid="stButton"] > button {{
        border-radius: 10px;
        border: none;
        font-weight: 600;
        color: white;
        background: linear-gradient(90deg, #ff007f, #ff6600);
        transition: all 0.2s ease;
    }}
    button:hover {{
        transform: scale(1.05);
    }}

    /* Chat Messages */
    .user-msg {{
        color: #00ffff;
        font-weight: 600;
        margin: 5px 0;
    }}
    .bot-msg {{
        color: #ffd700;
        font-weight: 600;
        margin: 5px 0;
    }}

    hr {{
        border: 1px solid rgba(255,255,255,0.3);
        margin: 20px 0;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# Set your background
set_background(r"C:\Users\Tharuni\Desktop\NIT\Nov month\3rd_Bert\bert chatbot\Gemini_Generated_Image_okxla4okxla4okxl.png")

# =========================================================
# INITIALIZE MODELS
# =========================================================
USE_LLM = True
LLM_MODEL = "microsoft/DialoGPT-small"

@st.cache_resource(show_spinner=False)
def load_models():
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    tokenizer, llm = None, None
    if USE_LLM:
        tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
        llm = AutoModelForCausalLM.from_pretrained(LLM_MODEL)
    return bert_model, tokenizer, llm

bert_model, tokenizer, llm = load_models()
recognizer = sr.Recognizer()
engine = pyttsx3.init()

# =========================================================
# KNOWLEDGE BASE
# =========================================================
qa_pairs = {
    # 👋 Greetings and Introductions
    "hi": "Hello there! How are you doing today?",
    "hello": "Hi! How can I assist you today?",
    "hey": "Hey there! What’s up?",
    "good morning": "Good morning! Hope you have a great day ahead!",
    "good afternoon": "Good afternoon! How’s your day going?",
    "good evening": "Good evening! What can I do for you?",
    "how are you": "I’m great, thanks for asking! How about you?",
    "who are you": "I’m an AI chatbot powered by BERT and LLMs!",
    "what is your name": "You can call me BERTy, your smart AI assistant!",
    "nice to meet you": "Nice to meet you too!",
    "thank you": "You’re welcome!",
    "thanks": "Happy to help!",
    "bye": "Goodbye! Take care!",
    "goodbye": "See you soon!",
    "see you later": "Sure! I’ll be right here when you come back!",

    # 🎵 Music Commands
    "play some music": "Sure! What song would you like me to play?",
    "play despacito": "Playing Despacito on YouTube!",
    "play a song": "Playing your favorite tune now!",
    "play relaxing music": "Here’s something calm and soothing for you.",
    "play party songs": "Here’s your party playlist! Let’s dance!",
    "play love songs": "Playing romantic hits now!",
    "play latest hits": "Fetching the latest trending songs for you!",

    # 🧠 General Knowledge
    "who is elon musk": "Elon Musk is the CEO of Tesla and SpaceX, and co-founder of OpenAI.",
    "who is the president of india": "As of 2025, the President of India is Droupadi Murmu.",
    "who invented the internet": "The Internet was invented by a group of scientists led by Vint Cerf and Bob Kahn.",
    "what is artificial intelligence": "Artificial Intelligence is the simulation of human intelligence in machines that can learn and solve problems.",
    "what is machine learning": "Machine learning is a subset of AI that enables systems to learn from data without being explicitly programmed.",
    "what is deep learning": "Deep learning uses neural networks with many layers to model complex patterns in data.",
    "what is bert": "BERT stands for Bidirectional Encoder Representations from Transformers, a model developed by Google for NLP.",
    "what is oci": "OCI stands for Oracle Cloud Infrastructure — a cloud computing platform by Oracle.",
    "who created you": "I was created by developers using BERT and modern NLP models!",
    "what language are you built in": "I’m built using Python and the Hugging Face Transformers library.",

    # 😂 Jokes and Fun
    "tell me a joke": "Why did the computer go to therapy? It had too many bytes of emotional baggage!",
    "another joke": "Why was the math book sad? Because it had too many problems.",
    "make me laugh": "I told my computer I needed a break, and it said ‘Why? You seem to be working fine!’",
    "funny joke": "Why did the programmer quit his job? Because he didn’t get arrays!",
    "tech joke": "Why do Java developers wear glasses? Because they don’t see sharp!",
    "dark joke": "I tried to catch some fog yesterday — I mist.",
    "ai joke": "I asked my neural network to tell a joke, but it overfitted to dad jokes!",

    # 🌐 Search and Info
    "search python basics": "Opening Google search for Python basics...",
    "search ai news": "Let me show you the latest AI news.",
    "search youtube": "Opening YouTube now.",
    "open google": "Opening Google search page for you.",
    "open youtube": "Sure! Opening YouTube now.",
    "search for latest tech news": "Here are the latest technology updates for you!",
    "search weather": "Let’s check today’s weather for you.",
    "what is the time": "Let me check the current time for you.",
    "what is the date": "Let me find out today’s date.",

    # 💬 Small Talk
    "what are you doing": "Just chatting with you and learning new things!",
    "do you sleep": "Nope, I’m awake 24/7 for you!",
    "do you have friends": "Yes! My friends are other AI models around the world!",
    "what’s your favorite color": "Blue — because it reminds me of the digital sky!",
    "what’s your favorite food": "I don’t eat, but I’ve heard binary cookies are great!",
    "do you like humans": "Of course! Humans are fascinating beings full of creativity!",
    "are you real": "As real as your imagination allows!",
    "do you feel emotions": "Not yet, but I can understand them through text!",
    "do you like music": "Absolutely! I can even play songs for you!",
    "can you dance": "Not physically, but my algorithms can groove to the rhythm!",

    # 💻 Tech / Programming
    "what is python": "Python is a high-level, interpreted programming language known for simplicity and versatility.",
    "what is java": "Java is an object-oriented programming language used in enterprise and Android development.",
    "what is c++": "C++ is a powerful programming language often used for system and game development.",
    "what is html": "HTML stands for HyperText Markup Language, used for creating web pages.",
    "what is css": "CSS stands for Cascading Style Sheets, used to style web pages.",
    "what is javascript": "JavaScript is a scripting language that makes web pages interactive.",
    "what is cloud computing": "Cloud computing allows storing and processing data on remote servers instead of local machines.",
    "what is data science": "Data Science is the study of extracting insights from data using statistics and algorithms.",
    "what is big data": "Big Data refers to massive datasets that are too large for traditional processing systems.",
    "what is blockchain": "Blockchain is a decentralized digital ledger for recording transactions securely.",

    # 💬 Emotions / Motivations
    "i am sad": "I’m sorry to hear that. Want to talk about it?",
    "i am happy": "That’s great! Keep smiling!",
    "i am bored": "Maybe I can cheer you up with a joke or a song?",
    "i am tired": "Take a break, you deserve it.",
    "i am stressed": "Try taking deep breaths — it really helps.",
    "i am angry": "It’s okay to feel angry sometimes. Do you want to vent?",
    "i am lonely": "I’m right here with you. You’re not alone.",
    "i am excited": "That’s awesome! Tell me what’s making you excited.",
    "i feel lazy": "Even machines need rest sometimes — it’s okay!",
    "i feel great": "That’s the spirit! Keep it up!",

    # ⚙️ System and Help
    "help": "Sure! You can ask me to play music, tell a joke, or answer questions.",
    "what can you do": "I can chat, search, play songs, tell jokes, and help you learn!",
    "commands": "You can try commands like ‘play music’, ‘tell me a joke’, or ‘search Python basics’.",
    "settings": "You can adjust my voice or enable continuous listening mode in settings.",
    "about": "I am a conversational AI chatbot powered by BERT and Transformer models.",
    "who built you": "I was created by developers passionate about NLP and AI.",
    "version": "You are chatting with version 2.5 of the BERT Chatbot.",

    # 🌎 General Conversations
    "how is the weather": "I can’t feel it, but I can fetch weather info if you’d like.",
    "where are you from": "I exist in the cloud — everywhere and nowhere!",
    "what’s your hobby": "Talking to interesting humans like you!",
    "do you like movies": "Yes! Especially sci-fi — I relate to robots.",
    "favorite movie": "I love The Matrix — it feels like home.",
    "do you like games": "Definitely! I’m good at logic puzzles and trivia.",
    "how old are you": "I’m as old as my last update!",
    "what day is it": "Let me check today’s date for you.",
    "can you sing": "I can hum in binary: 010101!",
    "can you code": "Of course! Python is my favorite language!"
}

# =========================================================
# UTILITIES
# =========================================================
def speak(text):
    """Speak text asynchronously (optional)."""
    def run_speech():
        try:
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            st.warning(f"Speech engine error: {e}")
    threading.Thread(target=run_speech, daemon=True).start()

def listen():
    """Voice input."""
    with sr.Microphone() as source:
        st.info("🎙️ Listening... Speak now!")
        audio = recognizer.listen(source)
        try:
            text = recognizer.recognize_google(audio)
            return text.lower()
        except:
            return "Sorry, I couldn’t understand you."

def bert_response(user_input):
    """Return the best BERT match."""
    corpus = list(qa_pairs.keys())
    corpus_embeddings = bert_model.encode(corpus, convert_to_tensor=True)
    query_embedding = bert_model.encode(user_input, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(query_embedding, corpus_embeddings)[0]
    best_match = torch.argmax(cos_scores).item()
    similarity = cos_scores[best_match].item()
    if similarity < 0.45:
        return None
    return qa_pairs[corpus[best_match]]

def llm_response(user_input, history):
    """Fallback to DialoGPT."""
    input_text = " ".join(history[-5:]) + " " + user_input
    inputs = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt')
    outputs = llm.generate(inputs, max_length=200, pad_token_id=tokenizer.eos_token_id)
    reply = tokenizer.decode(outputs[:, inputs.shape[-1]:][0], skip_special_tokens=True)
    return reply

# =========================================================
# STREAMLIT INTERFACE
# =========================================================
st.markdown("<h1>BERTConnect🤖</h1>", unsafe_allow_html=True)
st.markdown("<p>An interactive chatbot powered by BERT embeddings and conversational LLM intelligence.</p>", unsafe_allow_html=True)

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("💬 Type your message:")

col1, col2, col3 = st.columns(3)
with col1:
    speak_btn = st.button("🎙️ Speak")
with col2:
    send_btn = st.button("🚀 Send")
with col3:
    clear_btn = st.button("🧹 Clear Chat")

if speak_btn:
    user_input = listen()
    st.write(f"🗣️ You said: **{user_input}**")

if send_btn and user_input.strip():
    st.session_state.history.append(f"User: {user_input}")
    bot_reply = bert_response(user_input)

    if not bot_reply and USE_LLM:
        bot_reply = llm_response(user_input, st.session_state.history)
    elif not bot_reply:
        bot_reply = "I'm not sure how to respond to that."

    st.session_state.history.append(f"Bot: {bot_reply}")
    # speak(bot_reply)  # removed (no sound)

if clear_btn:
    st.session_state.history = []

st.markdown("<hr>", unsafe_allow_html=True)

for message in st.session_state.history[-10:]:
    if message.startswith("User:"):
        st.markdown(f"<div class='user-msg'>{message}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>{message}</div>", unsafe_allow_html=True)
