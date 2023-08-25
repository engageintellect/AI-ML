from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

# Sample data for fine-tuning
train_texts = [["Hello,", "my", "name", "is", "John."], ["I", "work", "at", "BTS."], ["My", "favorite", "color", "is", "blue."]]

# Define label mapping
label_map = {"O": 0, "B-PER": 1, "I-PER": 2, "B-ORG": 3}

# Initialize train_labels using integer labels
train_labels = [[label_map[label] for label in sentence_labels] for sentence_labels in [["O", "O", "O", "O", "B-PER"], ["O", "O", "O", "B-ORG"], ["O", "O", "O", "O", "O"]]]

# Tokenize the texts and align the labels
train_encodings = tokenizer(train_texts, truncation=True, padding=True, is_split_into_words=True)

# Print to see what's inside train_encodings
print(train_encodings.data)

train_labels_aligned = []

# Use the length of train_texts to loop through each sample
for i in range(len(train_texts)):
    try:
        word_ids = train_encodings.word_ids(batch_index=i)
        print(f'Working on sentence {i}, word IDs: {word_ids}')
    except IndexError as e:
        print(f'An index error occurred at sentence {i}: {e}')

for i in range(len(train_texts)):
    word_ids = train_encodings.word_ids(batch_index=i)
    prev_word_idx = None
    label_ids = []
    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != prev_word_idx:
            label_ids.append(train_labels[i][word_idx])
        else:
            label_ids.append(-100)
        prev_word_idx = word_idx
    
    train_labels_aligned.append(label_ids)

# Prepare the dataset
class NERDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = NERDataset(train_encodings.data, train_labels_aligned)

# Initialize the Trainer
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tuning
trainer.train()

# Save model and tokenizer
model.save_pretrained("./saved_model_directory")
tokenizer.save_pretrained("./saved_model_directory")
