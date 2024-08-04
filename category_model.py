import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import os

# Veri setini yükleyin
data = pd.read_csv('model_egitim_veri.csv')

# Metin ve etiket sütunlarını seçin
title_texts = data['temizlenmis_title'].tolist()
description_texts = data['temizlenmis_description'].tolist()
category_labels = data['category'].tolist()

# Tokenizer'ı yükleyin
tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')

# Tokenizasyon ve encoding işlemleri
max_len = 256

def encode_text(texts):
    encodings = tokenizer(
        texts,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='np'
    )
    return encodings

# Her sütunu tokenize edin
title_encodings = encode_text(title_texts)
description_encodings = encode_text(description_texts)

# TensorFlow tensor'ları oluşturun
input_ids = np.concatenate([title_encodings['input_ids'], description_encodings['input_ids']], axis=1)
attention_masks = np.concatenate([title_encodings['attention_mask'], description_encodings['attention_mask']], axis=1)

# Kategorik etiketleri sayısal değerlere dönüştürün
label_encoder = LabelEncoder()
category_labels = label_encoder.fit_transform(category_labels)

# Veriyi train, validation ve test olarak ayırın (70/20/10 oranında)
train_inputs, temp_inputs, train_labels, temp_labels = train_test_split(input_ids, category_labels, test_size=0.3, random_state=42)
val_inputs, test_inputs, val_labels, test_labels = train_test_split(temp_inputs, temp_labels, test_size=1/3, random_state=42)
train_masks, temp_masks = train_test_split(attention_masks, test_size=0.3, random_state=42)
val_masks, test_masks = train_test_split(temp_masks, test_size=1/3, random_state=42)

# DataLoader'ları oluşturun
batch_size = 16
train_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': train_inputs, 'attention_mask': train_masks}, train_labels)).shuffle(len(train_inputs)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': val_inputs, 'attention_mask': val_masks}, val_labels)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(({'input_ids': test_inputs, 'attention_mask': test_masks}, test_labels)).batch(batch_size)

def create_bert_cnn_model(n_classes):
    # BERT modeli
    bert_model = TFBertModel.from_pretrained('dbmdz/bert-base-turkish-cased')

    # Girdi katmanları
    input_ids = tf.keras.Input(shape=(2 * max_len,), dtype=tf.int32, name="input_ids")
    attention_mask = tf.keras.Input(shape=(2 * max_len,), dtype=tf.int32, name="attention_mask")

    # BERT modelinden çıkan son katman
    bert_outputs = bert_model(input_ids, attention_mask=attention_mask)
    sequence_output = bert_outputs.last_hidden_state

    # CNN katmanları
    cnn_output = tf.keras.layers.Conv1D(256, 3, activation='relu')(sequence_output)
    cnn_output = tf.keras.layers.GlobalMaxPooling1D()(cnn_output)

    # Fully connected katman
    output = tf.keras.layers.Dense(n_classes, activation='softmax')(cnn_output)

    # Modeli oluştur
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)

    # BERT modelinin tüm katmanlarını eğitilebilir hale getir
    for layer in bert_model.layers:
        layer.trainable = True

    return model

# Modeli ve optimizer'ı tanımlayın
n_classes = len(label_encoder.classes_)  # Kategori sayısı
model = create_bert_cnn_model(n_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Callback'leri tanımlayın
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Eğitimi başlat
epochs = 3
history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=[early_stopping])

# Modeli değerlendirin
test_predictions = model.predict(test_dataset)
test_predictions = np.argmax(test_predictions, axis=1)
test_labels = np.concatenate([y for x, y in test_dataset], axis=0)

# Sonuçları yazdırın
accuracy = accuracy_score(test_labels, test_predictions)
report = classification_report(test_labels, test_predictions, labels=np.arange(n_classes), target_names=label_encoder.classes_)

# Sonuçları dosyaya kaydedin
results_dir = 'model_results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

results_path = os.path.join(results_dir, 'category_results.txt')
with open(results_path, 'w') as f:
    f.write(f'Test Accuracy: {accuracy}\n')
    f.write(report)

print(f'Test Accuracy: {accuracy}')
print(report)

# Modeli kaydetmek için dizini oluştur
model_save_dir = 'saved_model'
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

# Modeli .h5 formatında kaydet
model_save_path_h5 = os.path.join(model_save_dir, 'category_model.h5')
model.save(model_save_path_h5, save_format='h5')

print(f"Model saved to {model_save_path_h5}")
