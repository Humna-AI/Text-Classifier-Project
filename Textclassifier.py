import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Input, Bidirectional, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN, LearningRateScheduler ,ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, TFBertModel, TFDistilBertModel, optimization_tf
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Lambda, SpatialDropout1D


# Load and preprocess data
with open('data/data.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]
df = pd.DataFrame(data)
df_english = df[df['language'] == 'English']
texts = df_english['text'].values
labels = df_english['labels'].values

# Label encoding and stratified split
label_map = {label: idx for idx, label in enumerate(np.unique(labels))}
labels_encoded = np.array([label_map[label] for label in labels])
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
)

# Class weights for imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# ======================
# 1. HIGH-ACCURACY LSTM
# ======================
# Tokenization and padding
vocab_size = 20000
max_length = 250
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)
X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')

# Load GloVe embeddings
embedding_dim = 300
embedding_index = {}
with open('data/glove.6B.300d.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = vector

# Create embedding matrix
embedding_matrix = np.zeros((vocab_size, embedding_dim))
word_index = tokenizer.word_index
for word, i in word_index.items():
    if i < vocab_size:
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Build LSTM Model
def build_model(trainable_embedding=False):
    model = Sequential([
        Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            weights=[embedding_matrix],
            input_length=max_length,
            trainable=trainable_embedding
        ),
        SpatialDropout1D(0.3),
        Bidirectional(LSTM(128, return_sequences=True)),
        BatchNormalization(),
        Bidirectional(LSTM(64)),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(len(label_map), activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Phase 1: Train with frozen embeddings
model = build_model(trainable_embedding=False)
model.fit(
    X_train_pad, y_train,
    epochs=40,
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        EarlyStopping(patience=30, restore_best_weights=True),
        TerminateOnNaN(),
        ModelCheckpoint('phase1_lstm.h5', save_best_only=True, monitor='val_loss')
    ],
    class_weight=class_weights_dict
)

# Load weights from Phase 1
model.load_weights('phase1_lstm.h5')

# Set embedding layer trainable for fine-tuning
model.layers[0].trainable = True

# Recompile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Phase 2: Fine-tune with unfrozen embeddings
model.fit(
    X_train_pad, y_train,
    epochs=20,  
    batch_size=32,
    validation_split=0.2,
    callbacks=[
        EarlyStopping(patience=10, restore_best_weights=True),
        TerminateOnNaN(),
        ModelCheckpoint('phase2_lstm.h5', save_best_only=True, monitor='val_loss')
    ],
    class_weight=class_weights_dict
)

# Evaluation
lstm_eval = model.evaluate(X_test_pad, y_test)
print(f"\nLSTM Results - Loss: {lstm_eval[0]:.4f}, Accuracy: {lstm_eval[1]:.4f}")

# Predictions
lstm_preds = np.argmax(model.predict(X_test_pad), axis=1)
reverse_label_map = {v: k for k, v in label_map.items()}

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, lstm_preds, target_names=label_map.keys()))

# ======================
# 2. OPTIMIZED TRANSFORMER
# ======================
max_length = 250  # Ensure consistency

bert_tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')

def encode_texts(texts, tokenizer, max_length=250):
    return tokenizer(
        texts.tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

train_encodings = encode_texts(X_train, bert_tokenizer, max_length=max_length)
test_encodings = encode_texts(X_test, bert_tokenizer, max_length=max_length)

transformer_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

# Define input layers with explicit names
input_ids = Input(shape=(max_length,), dtype=tf.int32, name="input_ids")
attention_mask = Input(shape=(max_length,), dtype=tf.int32, name="attention_mask")

# Transformer output
transformer_output = Lambda(
    lambda x: transformer_model(input_ids=x[0], attention_mask=x[1]).last_hidden_state[:, 0, :],
    output_shape=(768,)  # DistilBERT's hidden size is typically 768
)([input_ids, attention_mask])

# Add additional layers
x = BatchNormalization()(transformer_output)
x = Dense(256, activation='gelu')(x)
x = Dropout(0.2)(x)
output = Dense(len(label_map), activation='softmax')(x)

# Define the model
model = Model(inputs=[input_ids, attention_mask], outputs=output)

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=5e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    {'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask']},
    y_train,
    epochs=15,
    batch_size=8,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=2, restore_best_weights=True)]
)
# ======================
# EVALUATION
# ======================
transformer_eval = model.evaluate(
   {'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask']},
   y_test
)
print(f"Transformer Results - Loss: {transformer_eval[0]:.4f}, Accuracy: {transformer_eval[1]:.4f}")

# Save predictions
transformer_preds = np.argmax(model.predict(
   {'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask']}
), axis=1)

pd.DataFrame({'text': X_test, 'true': y_test, 'lstm_pred': lstm_preds}).to_csv('outputs/lstm_predictions.csv', index=False)
pd.DataFrame({'text': X_test, 'true': y_test, 'transformer_pred': transformer_preds}).to_csv('outputs/transformer_predictions.csv', index=False)

with open('outputs/lstm_metrics.txt', 'w') as f:
    f.write(f"Loss: {lstm_eval[0]}, Accuracy: {lstm_eval[1]}")
with open('outputs/transformer_metrics.txt', 'w') as f:
  f.write(f"Loss: {transformer_eval[0]}, Accuracy: {transformer_eval[1]}")

# Qualitative Analysis (print sample predictions)
reverse_label_map = {v: k for k, v in label_map.items()}
for i in range(10):
    print(f"Text: {X_test[i][:100]}...")
    print(f"True Label: {reverse_label_map[y_test[i]]}")
    print(f"LSTM Prediction: {reverse_label_map[lstm_preds[i]]}")
    print(f"Transformer Prediction: {reverse_label_map[transformer_preds[i]]}\n")

