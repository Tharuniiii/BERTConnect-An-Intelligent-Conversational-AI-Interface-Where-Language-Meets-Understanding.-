# BERTConnect-An-Intelligent-Conversational-AI-Interface-Where-Language-Meets-Understanding.-
BERTConnect is an AI-powered chatbot built with BERT embeddings and transformer-based conversational intelligence. It combines the contextual understanding power of BERT with the natural dialogue flow of LLMs, allowing users to interact through text or voice in a simple, elegant Streamlit interface. 

# ✨ Key Features

✅ Conversational Intelligence – Understands user intent using BERT’s contextual embeddings.
✅ LLM-Powered Dialogue – Uses transformer-based models like DialoGPT for natural replies.
✅ Speech-to-Text Input – Converts your voice into commands using SpeechRecognition.
✅ Text-to-Speech Output – AI speaks back to you using pyttsx3.
✅ Minimal Modern UI – Built with Streamlit and styled using inline CSS.
✅ Knowledge Base Integration – Quickly answers basic questions using semantic search.
✅ Offline-friendly Base Mode – Works with BERT embeddings even without internet.

# 🧩 Tech Stack
```
| Category                    | Technology Used                                                                                                    |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| **Frontend / UI**           | [Streamlit](https://streamlit.io/)                                                                                 |
| **Language Model**          | [BERT (SentenceTransformer)](https://www.sbert.net/), [DialoGPT](https://huggingface.co/microsoft/DialoGPT-medium) |
| **Speech Recognition**      | [SpeechRecognition](https://pypi.org/project/SpeechRecognition/)                                                   |
| **Text-to-Speech**          | [pyttsx3](https://pypi.org/project/pyttsx3/)                                                                       |
| **Backend Framework**       | Python 3.10+                                                                                                       |
| **Deep Learning Framework** | [PyTorch](https://pytorch.org/)                                                                                    |
| **Embeddings Similarity**   | [SentenceTransformers util](https://www.sbert.net/docs/package_reference/util.html)                                |
| **Deployment**              | Localhost / Streamlit Cloud                                                                                        |
```

# 🧠 How It Works
User Input (Text or Voice):
The user types or speaks a message. Speech input is transcribed using Google Speech Recognition API.

Intent Understanding:
The input is compared with a small knowledge base using BERT embeddings similarity (cosine similarity).

Response Generation:

If a matching intent is found → Responds directly using predefined QA pairs.

Otherwise → Uses DialoGPT to generate a contextual conversational response.

Text-to-Speech Output:
The chatbot speaks back using pyttsx3, creating a complete interactive loop.

# 🧱 Project Structure
```
BERTConnect/
│
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── assets/
│   └── background.png      # Background image for Streamlit UI
└── README.md               # Project documentation
```

# 🧾 Sample Interaction

You: Hi!
BERTConnect 🤖: Hello there! How can I help you today?

You: What is BERT?
BERTConnect 🤖: BERT stands for Bidirectional Encoder Representations from Transformers — it understands language context in both directions.

You: Tell me a joke!
BERTConnect 🤖: Why do programmers prefer dark mode? Because light attracts bugs!

🌈 User Interface Preview
<img width="1246" height="790" alt="Screenshot 2025-11-04 221101" src="https://github.com/user-attachments/assets/5b0ffbda-8ad4-4907-98d3-73547ddc13dd" />

# 🛠️ requirements.txt
```
streamlit
transformers
sentence-transformers
torch
pyttsx3
SpeechRecognition
pyaudio
```

