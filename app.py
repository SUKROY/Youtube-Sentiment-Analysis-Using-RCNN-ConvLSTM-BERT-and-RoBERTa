import streamlit as st
import torch
import torch.nn as nn
import pickle
import re
import string

from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    BertTokenizer,
    BertForSequenceClassification
)


# Memuat tokenizer dan model BERT yang telah
# dilatih sebelumnya dari folder model.

@st.cache_resource
def load_bert():

    tokenizer = BertTokenizer.from_pretrained(
        "models/bert_model"
    )

    model = BertForSequenceClassification.from_pretrained(
        "models/bert_model"
    )

    model.eval()

    return tokenizer, model

# Memuat tokenizer dan model RoBERTa yang telah
# dilatih sebelumnya dari folder model.

@st.cache_resource
def load_roberta():

    tokenizer = RobertaTokenizer.from_pretrained(
        "models/roberta_model"
    )

    model = RobertaForSequenceClassification.from_pretrained(
        "models/roberta_model"
    )

    model.eval()

    return tokenizer, model


# Mengatur judul halaman, ikon, dan tata letak.
st.set_page_config(
    page_title="YouTube Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

# Load Vocab
# Vocabulary digunakan oleh model RCNN dan
# ConvLSTM untuk mengubah kata menjadi angka.
@st.cache_resource
def load_vocab():
    with open("models/vocab.pkl", "rb") as f:
        return pickle.load(f)


# Load Label Encoder
# Digunakan untuk mengubah hasil prediksi
# numerik menjadi label sentimen.
@st.cache_resource
def load_label_encoder():
    with open("models/label_encoder.pkl", "rb") as f:
        return pickle.load(f)
    
vocab = load_vocab()
label_encoder = load_label_encoder()

# Parameter RCNN & ConvLSTM
MAX_LEN = 100
VOCAB_SIZE = len(vocab)
EMBED_DIM = 100
HIDDEN_DIM = 128
NUM_CLASSES = 3

# RCNN
# Embedding → BiLSTM → CNN → Pooling
# → Fully Connected Layer → Output
class RCNN(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.bilstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.conv = nn.Conv1d(
            hidden_dim * 2,
            128,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()

        self.pool = nn.AdaptiveMaxPool1d(1)

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(
            128,
            64
        )

        self.fc2 = nn.Linear(
            64,
            num_classes
        )

    def forward(self, x):

        x = self.embedding(x)

        lstm_out, _ = self.bilstm(x)

        x = lstm_out.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.squeeze(2)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


#  ConvLSTM
# Embedding → CNN → Pooling → LSTM
# → Fully Connected Layer → Output
class ConvLSTM(nn.Module):

    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embed_dim,
            padding_idx=0
        )

        self.conv1 = nn.Conv1d(
            embed_dim,
            128,
            kernel_size=5,
            padding=2
        )

        self.relu = nn.ReLU()

        self.pool = nn.MaxPool1d(2)

        self.lstm = nn.LSTM(
            128,
            hidden_dim,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(
            hidden_dim,
            64
        )

        self.fc2 = nn.Linear(
            64,
            num_classes
        )

    def forward(self, x):

        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        _, (hidden, _) = self.lstm(x)
        x = hidden[-1]
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


@st.cache_resource
def load_rcnn_model():

    model = RCNN(
        VOCAB_SIZE,
        EMBED_DIM,
        HIDDEN_DIM,
        NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(
            "models/rcnn_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model

@st.cache_resource
def load_convlstm_model():

    model = ConvLSTM(
        VOCAB_SIZE,
        EMBED_DIM,
        HIDDEN_DIM,
        NUM_CLASSES
    )

    model.load_state_dict(
        torch.load(
            "models/convlstm_model.pth",
            map_location="cpu"
        )
    )

    model.eval()

    return model


# Preprocessing
# Mengubah huruf menjadi lowercase,
# menghapus URL dan tanda baca.
def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    return text

#encoding
def encode_text(text):

    tokens = text.split()

    encoded = [
        vocab.get(
            token,
            vocab["<UNK>"]
        )
        for token in tokens
    ]

    if len(encoded) < MAX_LEN:

        encoded += [0] * (
            MAX_LEN - len(encoded)
        )

    else:

        encoded = encoded[:MAX_LEN]

    return torch.tensor(
        [encoded],
        dtype=torch.long
    )

# Menampilkan judul aplikasi dan pilihan model.
st.title(
    "YouTube Comment Sentiment Analysis"
)

model_choice = st.selectbox(
    "Choose Model",
    [
        "RCNN",
        "ConvLSTM",
        "BERT",
        "RoBERTa"
    ]
)

user_text = st.text_area(
    "Enter YouTube Comment"
)

# PROSES PREDIKSI
if st.button("Predict"):

    # INPUT KOMENTAR
    if user_text.strip() == "":

        st.warning(
            "Please enter text."
        )

    else:
        if model_choice == "RCNN":

            model = load_rcnn_model()

            text = preprocess_text(
                user_text
            )

            x = encode_text(text)

            with torch.no_grad():

                output = model(x)

                probs = torch.softmax(
                    output,
                    dim=1
                )

                pred = torch.argmax(
                    probs,
                    dim=1
                ).item()

        elif model_choice == "ConvLSTM":

            model = load_convlstm_model()

            text = preprocess_text(
                user_text
            )

            x = encode_text(text)

            with torch.no_grad():

                output = model(x)

                probs = torch.softmax(
                    output,
                    dim=1
                )

                pred = torch.argmax(
                    probs,
                    dim=1
                ).item()

        elif model_choice == "BERT":

            tokenizer, model = load_bert()

            inputs = tokenizer(
                user_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            )

            with torch.no_grad():

                output = model(**inputs)

                probs = torch.softmax(
                    output.logits,
                    dim=1
                )

                pred = torch.argmax(
                    probs,
                    dim=1
                ).item()

        else:

            tokenizer, model = load_roberta()

            inputs = tokenizer(
                user_text,
                return_tensors="pt",
                truncation=True,
                padding=True
            )

            with torch.no_grad():

                output = model(**inputs)

                probs = torch.softmax(
                    output.logits,
                    dim=1
                )

                pred = torch.argmax(
                    probs,
                    dim=1
                ).item()
        
        # Mengubah indeks hasil prediksi menjadi
        # label sentimen yang dapat dibaca pengguna.
        sentiment = label_encoder.inverse_transform(
            [pred]
        )[0]

        confidence = (
            probs.max().item() * 100
        )

        st.success(
            f"Sentiment: {sentiment}"
        )

        st.info(
            f"Confidence: {confidence:.2f}%"
        )

        st.write(
            f"Model Used: {model_choice}"
        )