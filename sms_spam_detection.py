"""
SMS Spam Detection using an LSTM (TensorFlow / Keras)

Run:
    python sms_spam_detection.py                # normal run
    python sms_spam_detection.py --compare      # also compare SimpleRNN vs LSTM vs GRU
    python sms_spam_detection.py --interactive  # afterwards, type your own SMS to classify

Put spam_sms.csv in the SAME folder as this script.
"""

import os
import re
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"   # hide TensorFlow's noisy info logs

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, SimpleRNN, LSTM, GRU, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, ConfusionMatrixDisplay)

# =====================================================================
# SETTINGS (all hyper-parameters in one place)
# =====================================================================
BASE_DIR    = Path(__file__).resolve().parent      # folder containing this script
FILE_PATH   = BASE_DIR / "spam_sms.csv"            # your dataset (same folder as this script)
SUBSET_SIZE = 2000                                 # ~2,000 labelled messages
VOCAB_SIZE  = 3000                                 # keep the 3000 most frequent words
EMBED_DIM   = 32                                   # size of each word vector
RNN_UNITS   = 64                                   # hidden units in the LSTM
BATCH_SIZE  = 32
EPOCHS      = 20                                   # upper limit; early stopping stops sooner
SEED        = 42

tf.keras.utils.set_random_seed(SEED)               # reproducible results


def header(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =====================================================================
# STEP 1: LOAD THE DATASET
# =====================================================================
header("STEP 1: Load dataset")

if not FILE_PATH.exists():
    sys.exit(f"ERROR: '{FILE_PATH}' not found.\n"
             "Place spam_sms.csv in the same folder as this script.")


def read_csv_safely(path):
    """Kaggle file is often latin-1 encoded: try utf-8 first, then latin-1."""
    for enc in ("utf-8", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read the file with utf-8 or latin-1")


raw = read_csv_safely(FILE_PATH)
print("Raw shape:", raw.shape)
print("Columns  :", list(raw.columns))

# find label and text columns (names differ between Kaggle versions: v1/v2, Category/Message ...)
cols = {c.lower().strip(): c for c in raw.columns}
label_col = next((cols[c] for c in ("v1", "label", "category", "class", "type") if c in cols), raw.columns[0])
text_col = next((cols[c] for c in ("v2", "message", "text", "sms", "sms_text") if c in cols), raw.columns[1])
print(f"Using '{label_col}' as label and '{text_col}' as text")

df = raw[[label_col, text_col]].copy()             # drop empty 'Unnamed' columns
df.columns = ["label", "text"]

# =====================================================================
# STEP 2: HANDLE MISSING VALUES AND CLEAN THE DATASET
# =====================================================================
header("STEP 2: Missing values and cleaning")

print("Missing values per column:")
print(df.isnull().sum(), "\n")

df = df.dropna(subset=["label", "text"])           # remove rows with missing values
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]                          # remove empty messages

before = len(df)
df = df.drop_duplicates(subset="text")             # duplicates would leak between train and test
print(f"Duplicates removed: {before - len(df)}")

# text label -> number  (ham = 0, spam = 1)
df["label"] = (df["label"].astype(str).str.lower().str.strip()
               .map({"ham": 0, "spam": 1, "0": 0, "1": 1}))
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

print("Clean dataset size:", len(df))

# stratified subset of ~2000 messages (keeps the same spam : ham ratio)
if len(df) > SUBSET_SIZE:
    df, _ = train_test_split(df, train_size=SUBSET_SIZE, stratify=df["label"], random_state=SEED)
df = df.reset_index(drop=True)

print("Subset size:", len(df))
print(df["label"].value_counts().rename({0: "ham", 1: "spam"}))
print(f"Spam percentage: {df['label'].mean() * 100:.1f}%  -> imbalanced data")

# =====================================================================
# STEP 3: PREPROCESS THE SMS TEXT
# =====================================================================
header("STEP 3: Text preprocessing")


def clean_text(text):
    text = text.lower()                                        # FREE == free
    text = re.sub(r"(https?://\S+|www\.\S+)", " url ", text)   # every link -> 'url'
    text = re.sub(r"[£$€]", " money ", text)                   # currency symbols -> 'money'
    text = re.sub(r"\d+", " num ", text)                       # every number -> 'num'
    text = re.sub(r"[^a-z\s]", " ", text)                      # keep letters only
    text = re.sub(r"\s+", " ", text).strip()                   # remove extra spaces
    return text
    # NOTE: stop-words are NOT removed - an LSTM uses word order ("text TO win")


df["clean"] = df["text"].apply(clean_text)

for i in range(3):
    print("ORIGINAL:", df["text"][i])
    print("CLEANED :", df["clean"][i], "\n")

# =====================================================================
# STEP 4: TRAIN / TEST SPLIT  (before tokenizing, to avoid data leakage)
# =====================================================================
header("STEP 4: Train / test split")

X_train, X_test, y_train, y_test = train_test_split(
    df["clean"], df["label"],
    test_size=0.20, stratify=df["label"], random_state=SEED)

y_train = y_train.values
y_test = y_test.values
print("Train:", len(X_train), "| Test:", len(X_test))
print("Spam in train:", y_train.sum(), "| Spam in test:", y_test.sum())

# =====================================================================
# STEP 5: TOKENIZATION AND PADDING
# =====================================================================
header("STEP 5: Tokenization and padding")

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")   # <OOV> = unseen word, id 0 = padding
tokenizer.fit_on_texts(X_train)                                  # vocabulary from TRAIN data only

train_seq = tokenizer.texts_to_sequences(X_train)                # words -> integer ids
test_seq = tokenizer.texts_to_sequences(X_test)

lengths = [len(s) for s in train_seq]
MAX_LEN = int(np.percentile(lengths, 95))                        # covers 95% of messages fully
print("Unique words in training data:", len(tokenizer.word_index))
print(f"Average length: {np.mean(lengths):.1f} | longest: {max(lengths)} | MAX_LEN (95th percentile): {MAX_LEN}")

X_train_pad = pad_sequences(train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_test_pad = pad_sequences(test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

print("\nExample message :", X_train.iloc[0])
print("As integer ids  :", train_seq[0])
print("After padding   :", X_train_pad[0])
print("Shape of training matrix:", X_train_pad.shape)

# =====================================================================
# STEP 6: BUILD THE MODEL
#   Simple RNN : vanishing gradient, forgets early words
#   LSTM       : gates + cell state remember important words over the sequence  <- chosen
#   GRU        : lighter alternative, similar accuracy
# =====================================================================
header("STEP 6: Build the LSTM model")


def build_model(kind="LSTM"):
    rnn_layer = {"SimpleRNN": SimpleRNN, "LSTM": LSTM, "GRU": GRU}[kind]
    model = Sequential([
        Input(shape=(MAX_LEN,)),
        Embedding(input_dim=VOCAB_SIZE, output_dim=EMBED_DIM, mask_zero=True),  # word id -> 32-d vector, padding ignored
        rnn_layer(RNN_UNITS),                                                   # reads the sequence -> one summary vector
        Dropout(0.3),                                                           # reduces overfitting
        Dense(1, activation="sigmoid"),                                         # P(spam) between 0 and 1
    ], name=f"{kind}_spam_classifier")
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


model = build_model("LSTM")
model.summary()

# =====================================================================
# STEP 7: TRAIN THE MODEL
# =====================================================================
header("STEP 7: Train")

# spam is only ~12% of the data -> give spam mistakes a bigger penalty
weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
class_weight = {0: float(weights[0]), 1: float(weights[1])}
print("Class weights:", class_weight)

early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

history = model.fit(
    X_train_pad, y_train,
    validation_split=0.10,          # 10% of training data used to watch for overfitting
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=2)

# training curves (shown at the end of the script)
fig1, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(history.history["loss"], label="train loss")
ax[0].plot(history.history["val_loss"], label="validation loss")
ax[0].set_title("Loss"); ax[0].set_xlabel("epoch"); ax[0].legend()
ax[1].plot(history.history["accuracy"], label="train accuracy")
ax[1].plot(history.history["val_accuracy"], label="validation accuracy")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("epoch"); ax[1].legend()
fig1.tight_layout()
fig1.savefig(BASE_DIR / "training_curves.png", dpi=150)

# =====================================================================
# STEP 8: EVALUATE  (accuracy, precision, recall, F1)
# =====================================================================
header("STEP 8: Evaluation on the unseen test set")

y_prob = model.predict(X_test_pad, verbose=0).ravel()    # probability of spam
y_pred = (y_prob >= 0.5).astype(int)                     # >= 0.5 -> spam

print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}\n")
print(classification_report(y_test, y_pred, target_names=["ham", "spam"], digits=4))

# =====================================================================
# STEP 9: CONFUSION MATRIX
# =====================================================================
header("STEP 9: Confusion matrix")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"TN={tn} (ham allowed)   FP={fp} (genuine SMS wrongly blocked)")
print(f"FN={fn} (spam missed)   TP={tp} (spam caught)")

fig2, ax2 = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"]).plot(
    ax=ax2, cmap="Blues", values_format="d", colorbar=False)
ax2.set_title("Confusion Matrix (test set)")
fig2.tight_layout()
fig2.savefig(BASE_DIR / "confusion_matrix.png", dpi=150)

# =====================================================================
# STEP 10: PREDICTIONS ON UNSEEN SMS MESSAGES
#   Same pipeline as training: clean -> tokenize (FITTED tokenizer) -> pad -> predict
# =====================================================================
header("STEP 10: Predictions on unseen SMS messages")


def predict_sms(messages, threshold=0.5):
    cleaned = [clean_text(m) for m in messages]
    seqs = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(seqs, maxlen=MAX_LEN, padding="post", truncating="post")
    probs = model.predict(padded, verbose=0).ravel()
    for msg, p in zip(messages, probs):
        label = "SPAM" if p >= threshold else "HAM "
        print(f"[{label}]  P(spam) = {p:.3f}  |  {msg}")


new_messages = [
    "Congratulations! You have won a $1000 gift card. Click http://bit.ly/claim-now to claim your prize today",
    "URGENT! Your account has been suspended. Call 09061234567 immediately to verify",
    "Free entry in our weekly draw! Text WIN to 80082 to get a free ringtone",
    "You have been selected for a free holiday. Reply YES to claim now",
    "Hey, are we still meeting for lunch at 1pm today?",
    "Can you send me the notes from yesterday's class? I missed it",
    "Mom said dinner is ready, come home soon",
    "Happy birthday da! Party at my place tonight, don't be late",
]
predict_sms(new_messages)

# =====================================================================
# OPTIONAL: compare Simple RNN, LSTM and GRU (python sms_spam_detection.py --compare)
# =====================================================================
if "--compare" in sys.argv:
    header("OPTIONAL: Simple RNN vs LSTM vs GRU")
    rows = []
    for kind in ["SimpleRNN", "LSTM", "GRU"]:
        tf.keras.utils.set_random_seed(SEED)
        m = build_model(kind)
        m.fit(X_train_pad, y_train, validation_split=0.10, epochs=EPOCHS, batch_size=BATCH_SIZE,
              class_weight=class_weight, verbose=0,
              callbacks=[EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)])
        p = (m.predict(X_test_pad, verbose=0).ravel() >= 0.5).astype(int)
        rows.append({"model": kind, "parameters": m.count_params(),
                     "accuracy": accuracy_score(y_test, p), "precision": precision_score(y_test, p),
                     "recall": recall_score(y_test, p), "f1": f1_score(y_test, p)})
    print(pd.DataFrame(rows).round(4).to_string(index=False))

print(f"\nSaved plots: {BASE_DIR / 'training_curves.png'} and {BASE_DIR / 'confusion_matrix.png'}")

# show both plots (close the plot windows to continue)
plt.show()

# =====================================================================
# OPTIONAL: type your own SMS (python sms_spam_detection.py --interactive)
# =====================================================================
if "--interactive" in sys.argv:
    header("INTERACTIVE MODE (press Enter on an empty line to quit)")
    while True:
        msg = input("\nType an SMS: ").strip()
        if not msg:
            break
        predict_sms([msg])