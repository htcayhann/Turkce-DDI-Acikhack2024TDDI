import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


data = pd.read_csv('model_egitim_veri.csv')


title_texts = data['temizlenmis_title'].tolist()
description_texts = data['temizlenmis_description'].tolist()
labels = data['Real/Fake'].tolist()


tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')


max_len = 256

def encode_text(texts):
   
    encodings = tokenizer(texts,
                          max_length=max_len,
                          padding='max_length',
                          truncation=True,
                          return_attention_mask=True,
                          return_tensors='np')  # NumPy dizisi olarak döndür
    return encodings


title_encodings = encode_text(title_texts)
description_encodings = encode_text(description_texts)

input_ids = np.concatenate([title_encodings['input_ids'], description_encodings['input_ids']], axis=1)
attention_masks = np.concatenate([title_encodings['attention_mask'], description_encodings['attention_mask']], axis=1)
labels = np.array(labels)

train_inputs, val_inputs, train_labels, val_labels = train_test_split(input_ids, labels, test_size=0.2)
train_masks, val_masks = train_test_split(attention_masks, test_size=0.2)

train_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': train_inputs, 'attention_mask': train_masks}, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': val_inputs, 'attention_mask': val_masks}, val_labels))

train_dataset = train_dataset.shuffle(len(train_inputs)).batch(16)
val_dataset = val_dataset.batch(16)

def create_bert_cnn_model(n_classes):
    # BERT modeli
    bert_model = TFBertModel.from_pretrained('dbmdz/bert-base-turkish-cased')

    input_ids = tf.keras.Input(shape=(2 * max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(2 * max_len,), dtype=tf.int32, name="attention_mask")

    bert_outputs = bert_model(input_ids, attention_mask=attention_mask)
    sequence_output = bert_outputs.last_hidden_state

    # CNN katmanları
    cnn_output = tf.keras.layers.Conv1D(256, 3, activation='relu')(sequence_output)
    cnn_output = tf.keras.layers.GlobalMaxPooling1D()(cnn_output)

    output = tf.keras.layers.Dense(n_classes, activation='softmax')(cnn_output)

    # Modeli oluşturulması
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    return model

n_classes = 2  # İki sınıflı sınıflandırma
model = create_bert_cnn_model(n_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',   # İzlenecek değer
    patience=2,           # İyileşme olmayan epoch sayısı
    restore_best_weights=True # En iyi ağırlıkları geri yükle
)

# Eğitim
epochs = 3
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=[early_stopping] # EarlyStopping callback'ini ekleyin
)

# Modeli değerlendirmesi
val_predictions = model.predict(val_dataset)
val_predictions = np.argmax(val_predictions, axis=1)
val_labels = np.concatenate([y for x, y in val_dataset], axis=0)


print(f'Validation Accuracy: {accuracy_score(val_labels, val_predictions)}')
print(classification_report(val_labels, val_predictions))


model.save('bert_cnn_model.h5')
