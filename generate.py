from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import sys
import torch

# Windows consoles default to a code page that cannot print Turkish characters
sys.stdout.reconfigure(encoding="utf-8")

model_path = "./final_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

sample_text = """
Yapay zeka teknolojileri son yıllarda büyük bir ivme kazandı. Özellikle doğal dil işleme alanındaki
gelişmeler, makinelerin insan dilini anlamasını ve üretmesini mümkün kılıyor. Uzmanlar, önümüzdeki
on yıl içinde yapay zekanın eğitim, sağlık ve finans sektörlerinde devrim yaratacağını öngörüyor.
Ancak bu hızlı gelişim, veri gizliliği ve etik tartışmalarını da beraberinde getiriyor. Şirketler
ve hükümetler, yapay zekanın güvenli kullanımı için yasal düzenlemeler üzerinde çalışmaya başladı.
"""

inputs = tokenizer(sample_text, return_tensors="pt", max_length=512, truncation=True).to(device)

outputs = model.generate(
    **inputs,
    max_length=128,
    num_beams=4,
    no_repeat_ngram_size=3,
    repetition_penalty=1.2,
    early_stopping=True
)

summary = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Original Text:")
print(sample_text.strip())
print("-" * 50)
print("Generated Summary:")
print(summary)
