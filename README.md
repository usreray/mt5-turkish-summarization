# Turkish Text Summarization with mT5

Fine-tuning [`google/mt5-small`](https://huggingface.co/google/mt5-small) for abstractive summarization of Turkish news articles, using the Turkish subset of the [MLSUM](https://huggingface.co/datasets/reciTAL/mlsum) dataset and the Hugging Face `Seq2SeqTrainer`.

## Project structure

| File | Purpose |
| --- | --- |
| `train.py` | Loads MLSUM Turkish, tokenizes it, fine-tunes mT5-small and saves the model to `final_model/` |
| `generate.py` | Loads the fine-tuned model and summarizes a sample Turkish text |
| `requirements.txt` | Python dependencies |

Model weights are not stored in this repository because of their size (about 2.2 GB).

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Linux / macOS
```

Install PyTorch for your platform and CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/), then:

```bash
pip install -r requirements.txt
```

## Training

```bash
python train.py
```

| Setting | Value |
| --- | --- |
| Base model | `google/mt5-small` |
| Dataset | MLSUM Turkish (`train` / `validation` splits) |
| Max input / summary length | 512 / 128 tokens |
| Learning rate | 3e-5 |
| Batch size | 16 per device, gradient accumulation 2 (effective 32) |
| Epochs | 3 |
| Precision | bf16 |
| Hardware | Google Colab, NVIDIA H100 |

| MLSUM Turkish split | Articles |
| --- | --- |
| Train | 249,277 |
| Validation | 11,565 |
| Test | 12,775 |

bf16 is used because training with fp16 produced `NaN` losses. This is a known problem with T5-family models: their activations can overflow the fp16 range, while bf16 has the same range as fp32. Training with `bf16=True` requires a GPU with bfloat16 support (NVIDIA Ampere or newer). On older GPUs, set `bf16=False` and `fp16=True` in `train.py`. Checkpoints are written to `mt5_turkish_summarization_run/` every 2000 steps, and the final model is saved to `final_model/`.

## Generating a summary

```bash
python generate.py
```

`generate.py` uses beam search (`num_beams=4`, `no_repeat_ngram_size=3`, `repetition_penalty=1.2`). Example output of the trained model:

**Input**

> Yapay zeka teknolojileri son yıllarda büyük bir ivme kazandı. Özellikle doğal dil işleme alanındaki gelişmeler, makinelerin insan dilini anlamasını ve üretmesini mümkün kılıyor. Uzmanlar, önümüzdeki on yıl içinde yapay zekanın eğitim, sağlık ve finans sektörlerinde devrim yaratacağını öngörüyor. Ancak bu hızlı gelişim, veri gizliliği ve etik tartışmalarını da beraberinde getiriyor. Şirketler ve hükümetler, yapay zekanın güvenli kullanımı için yasal düzenlemeler üzerinde çalışmaya başladı.

**Summary**

> Yapay zeka teknolojileri son yıllarda büyük bir ivme kazandı. Özellikle doğal dil işleme alanındaki gelişmeler, makinelerin insan dilini anlamasını ve üretmesini mümkün kılıyor.

The summary is the first two sentences of the input, copied word for word.

## Why the model copies the first sentences

The model has a strong **lead bias**: instead of writing new sentences (abstractive summarization), it usually copies the opening sentences of the article (extractive behavior). Three factors contribute to this.

**1. The training data rewards the lead.** In MLSUM news, the most important information comes first, and the reference summaries follow the article's opening closely. Measured on the MLSUM Turkish validation split (9,437 articles with at least 6 sentences):

| Share of reference summary words that appear in... | |
| --- | --- |
| the first 2 sentences of the article | 42.9% |
| 2 randomly chosen later sentences | 18.1% |
| nowhere in the article (new words) | 30.1% |

The opening sentences cover about 2.4 times more of the summary than other sentences, so starting from the lead is a reliable way to reduce the training loss.

**2. Beam search prefers safe, high-probability text.** Copying from the input is the most likely continuation for a seq2seq model, and beam search (`num_beams=4`) picks the most likely sequence. The reference summaries still contain about 30% new words, but the generated summaries contain almost none, so the model is more extractive than the data it was trained on.

**3. Model size (likely).** mT5-small has about 300M parameters, and most of them belong to the 250k-token multilingual vocabulary. Small models are known to rewrite less than larger ones. This project did not train a larger model, so the effect of size was not measured directly.

The `repetition_penalty` and `no_repeat_ngram_size` settings in `generate.py` do not reduce lead bias. They only penalize repeating words that are already in the generated summary, not copying words from the article.

### Meaning errors under forced generation

In an earlier experiment with a different generation setting, the model shortened *"Yetkililer, kullanıcıları ürünleri derhal kullanmayı bırakmaları konusunda uyardı"* (warned users to **stop using** the products) to *"... kullanmaları konusunda uyardı"* (warned users to **use** them). The model dropped the verb *bırakmak* ("to stop"), which reversed the meaning. The *-ma* in *kullanmayı* is a verbal-noun suffix, not a negation suffix, so this is a dropped-word error rather than a negation error. It shows that a small model can lose critical meaning when it is pushed away from copying. With the settings in `generate.py`, the published model copies this sentence correctly.

## Notes

- `datasets` 4.x no longer supports dataset loading scripts, so `train.py` loads the Parquet conversion of MLSUM (`revision="refs/convert/parquet"`, `data_dir="tu"`).
- When loading `final_model/` with recent `transformers` versions, a warning about tied weights may appear. It is expected: the checkpoint stores the input embeddings and the output layer separately, and `transformers` keeps them untied.

## Dataset license

MLSUM is released for research purposes. See the [dataset card](https://huggingface.co/datasets/reciTAL/mlsum) and the paper: Scialom et al., *MLSUM: The Multilingual Summarization Corpus*, EMNLP 2020. Models trained on it should be used accordingly.
