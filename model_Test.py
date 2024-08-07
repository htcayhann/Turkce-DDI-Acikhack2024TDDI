import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from sklearn.preprocessing import LabelEncoder
import random


model_save_path_h5 = 'bert_cnn_model.h5'
model = tf.keras.models.load_model(model_save_path_h5, custom_objects={'TFBertModel': TFBertModel})

# Veri seti
data = pd.read_csv('model_egitim_veri.csv')

k = 100
random_indices = random.sample(range(len(data)), k)
sample_data = data.iloc[random_indices]

title_texts = sample_data['temizlenmis_title'].tolist()
description_texts = sample_data['temizlenmis_description'].tolist()
category_labels = sample_data['Real/Fake'].tolist()

tokenizer = BertTokenizer.from_pretrained('dbmdz/bert-base-turkish-cased')

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


title_encodings = encode_text(title_texts)
description_encodings = encode_text(description_texts)

input_ids = np.concatenate([title_encodings['input_ids'], description_encodings['input_ids']], axis=1)
attention_masks = np.concatenate([title_encodings['attention_mask'], description_encodings['attention_mask']], axis=1)

label_encoder = LabelEncoder()
category_labels_encoded = label_encoder.fit_transform(category_labels)

test_dataset = tf.data.Dataset.from_tensor_slices(
    ({'input_ids': input_ids, 'attention_mask': attention_masks}, category_labels_encoded)).batch(16)

# Model test 
test_predictions = model.predict(test_dataset)
test_predictions = np.argmax(test_predictions, axis=1)


results_file = 'haber_test_results_1.txt'
with open(results_file, 'w', encoding='utf-8') as f:
    correct_count = 0
    for i, (input_id, true_label, pred) in enumerate(zip(input_ids, category_labels_encoded, test_predictions)):
        predicted_label = label_encoder.inverse_transform([pred])[0]
        true_label_text = label_encoder.inverse_transform([true_label])[0]
        if predicted_label == true_label_text:
            correct_count += 1
        f.write(f"Sample {i + 1}:\n")
        f.write(f"Title: {title_texts[i]}\n")
        f.write(f"Description: {description_texts[i]}\n")
        f.write(f"True Label: {true_label_text}\n")
        f.write(f"Predicted Label: {predicted_label}\n")
        f.write("=" * k + "\n")

    # Doğru tahminlerin sayısını ve doğruluğu yazdırın
    accuracy = correct_count / k
    f.write(f"Doğru Tahmin Sayısı: {correct_count}\n")
    f.write(f"Doğruluk: {accuracy}\n")

print(f"Sonuçlar '{results_file}' dosyasına kaydedildi.")
