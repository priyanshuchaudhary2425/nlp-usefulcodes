# This code is for token classification this is for refrence dont just copy and paste!!

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels




def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs




tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)




# Use this if you are working with SequenceClassification This is for refrence dont just copy and paste!

def tokenize_data(examples):
    # Tokenize the text
    tokenized_inputs = tokenizer(examples["text"], truncation=True, padding=True)

    # Extract the labels
    labels = examples["label"]

    return {"input_ids": tokenized_inputs["input_ids"], "attention_mask": tokenized_inputs["attention_mask"], "labels": labels}

# Tokenize the datasets
tokenized_datasets = raw_datasets.map(
    tokenize_data,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)
