# Use this to leverage ease of huggingface train args and train
make sure to use 

# pip install transformers -U       at the beginning because this will restart the notebook

from transformers import TrainingArguments

args = TrainingArguments(
    "bert-finetuned-ner",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    num_train_epochs=3,
    weight_decay=0.01,
    # push_to_hub=True,  # Optional if you have to push to hub
)


from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)
trainer.train()


# trainer.push_to_hub(commit_message="Training complete")    optional!
