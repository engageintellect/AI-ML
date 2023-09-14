from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import torch

def create_training_data(sentences, labels):
    label_set = set(label for sentence_labels in labels for label in sentence_labels)
    label_map = {label: i for i, label in enumerate(label_set)}
    
    train_texts = []
    train_labels = []
    
    for sentence, sentence_labels in zip(sentences, labels):
        tokenized_sentence = sentence.split(" ")
        train_texts.append(tokenized_sentence)

        integer_labels = [label_map[label] for label in sentence_labels]
        train_labels.append(integer_labels)
        
    return train_texts, train_labels, label_map

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")

with open('data.json') as f:
    data = json.load(f)

sentences = [item['sentence'] for item in data]
labels = [item['labels'] for item in data]

# Now you can pass sentences and labels to your function 
train_texts, train_labels, label_map = create_training_data(sentences, labels)

# Explicitly define the label mapping with all 9 original labels
label_map = {
    "O": 0, 
    "B-PER": 1, 
    "I-PER": 2, 
    "B-ORG": 3, 
    "I-ORG": 4,
    "B-LOC": 5, 
    "I-LOC": 6,
    "B-MISC": 7, 
    "I-MISC": 8
}

# Update the model's config for the number of labels
model.config.num_labels = len(label_map)

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
    num_train_epochs=5,
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
