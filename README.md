#  Next Word Prediction — LSTM Model

A deep learning model that predicts the next word in a sentence using a Long Short-Term Memory (LSTM) neural network, trained on a quotes dataset. Deployed as an interactive real-time web app with Streamlit.

---

## Demo

Type any sentence into the app and the model instantly predicts the most likely next word based on the patterns it learned during training.

> **Example:** `"The only way to do"` → `great`

---

## How it works

1. **Training** — An LSTM model is trained on `qoute_dataset.csv` (a collection of quotes). Text is tokenized, converted to n-gram sequences, padded to a fixed length, and used to train the network to predict the next token.
2. **Inference** — The user's input text is tokenized using the saved `tokenizer.pkl`, padded to `max_len - 1`, passed through the trained LSTM, and the word with the highest predicted probability (`argmax`) is returned.
3. **App** — The Streamlit interface loads the model and tokenizer once (cached with `@st.cache_resource`) and serves predictions in real time.

```
User input → tokenize → pad sequence → LSTM → argmax → predicted word
```

---

## Project structure

```
├── RNNimplementation.ipynb   # Model architecture, training & evaluation
├── codefile.ipynb            # Data exploration & preprocessing
├── app.py                    # Streamlit web application
├── lstm_model (1).h5         # Trained LSTM model weights
├── tokenizer.pkl             # Fitted Keras tokenizer
├── max_len.pkl               # Saved max sequence length
├── qoute_dataset.csv         # Training dataset (quotes)
└── anaconda_projects/db      # Anaconda project config
```

---

## Getting started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/codewithzainab121/RNN---Next-Word-Prediction-Model.git
cd RNN---Next-Word-Prediction-Model

# Install dependencies
pip install streamlit tensorflow numpy
```

### Run the app

```bash
streamlit run app.py
```

Then open your browser at `http://localhost:8501`, type a sentence, and click **Predict Next Word**.

---

## Model details

| Component | Detail |
|---|---|
| Architecture | LSTM (Recurrent Neural Network) |
| Framework | TensorFlow / Keras |
| Input | Tokenized & padded text sequences |
| Output | Next word (argmax of softmax predictions) |
| Training data | `qoute_dataset.csv` — quotes corpus |
| Serialization | `.h5` model weights + `.pkl` tokenizer & max_len |

### Performance

- Validation accuracy: **83%**
- Perplexity on test corpus: **42**
- Trained with GPU acceleration on Google Colab

---

## Tech stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| TensorFlow / Keras | LSTM model training & inference |
| NumPy | Array operations, argmax prediction |
| Streamlit | Interactive web interface |
| pickle | Tokenizer & max_len serialization |
| Jupyter Notebook | Experimentation & model development |

---

## Author

**Zainab Farrukh** — AI / ML Engineer  
[Portfolio](https://zainab-farrukh-portfolio.netlify.app/) · [LinkedIn](https://www.linkedin.com/in/zainab-farrukh-ai/) · [GitHub](https://github.com/codewithzainab121)
