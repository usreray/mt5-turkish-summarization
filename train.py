from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq

print("Loading MLSUM Turkish dataset...")
# datasets 4.x no longer runs loading scripts, so use the Parquet conversion of MLSUM
dataset = load_dataset("reciTAL/mlsum", revision="refs/convert/parquet", data_dir="tu")

model_checkpoint = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)


def preprocess_function(examples):
    inputs = tokenizer(examples["text"], max_length=512, truncation=True)
    targets = tokenizer(text_target=examples["summary"], max_length=128, truncation=True)
    inputs["labels"] = targets["input_ids"]
    return inputs


tokenized_datasets = dataset.map(preprocess_function, batched=True, num_proc=4)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="./mt5_turkish_summarization_run",
    eval_strategy="steps",
    eval_steps=2000,
    save_strategy="steps",
    save_steps=2000,
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=False,
    bf16=True,  # requires a GPU with bfloat16 support (NVIDIA Ampere or newer)
    logging_steps=100,
    push_to_hub=False,
    report_to="none",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("./final_model")
print("Training complete. Model saved successfully.")
