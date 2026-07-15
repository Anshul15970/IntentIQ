# 🏦 IntentIQ

IntentIQ is an intelligent banking intent classification system that compares prompt engineering techniques with LoRA fine-tuning on the Banking77 dataset.

The project demonstrates how different approaches perform on the same intent classification task through an interactive Gradio application.

---

## Features

- 🔹 Zero-shot Prompting
- 🔹 Few-shot Prompting
- 🔹 Dynamic Few-shot Prompting
- 🔹 LoRA Fine-tuned Model
- 🔹 Compare predictions across all models
- 🔹 Benchmark dashboard
- 🔹 Confusion matrix visualization
- 🔹 Error analysis dashboard
- 🔹 Automatic project summary and recommendations

---

## Models Used

- Qwen2.5-0.5B-Instruct
- LoRA Fine-tuning (PEFT)

---

## Dataset

Banking77

https://huggingface.co/datasets/PolyAI/banking77

---

## Project Structure

```
IntentIQ/
│
├── app.py
├── benchmark/
├── config/
├── finetuning/
├── models/
├── prompts/
├── results/
├── utils/
├── visualization/
├── requirements.txt
└── README.md
```

---

## Running Locally

Clone the repository

```bash
git clone https://github.com/Anshul15970/IntentIQ
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python app.py
```

---

## Application Features

### Prediction
- Predict intent using any individual model.

### Compare Models
- Compare predictions from:
  - Zero-shot
  - Few-shot
  - Dynamic Few-shot
  - LoRA

### Benchmark Dashboard
Displays

- Accuracy
- Precision
- Recall
- F1 Score
- Average Inference Time

### Confusion Matrix
Visual comparison of prediction performance for every model.

### Error Analysis
Displays

- Top misclassification pairs
- Error summary
- Automatic analysis
- Overall recommendation

---

## Results

The LoRA fine-tuned model achieved the highest classification performance and significantly reduced semantic confusion compared to prompt-based approaches.

Dynamic Few-shot performed best among prompting methods.

---

## Technologies

- Python
- PyTorch
- Hugging Face Transformers
- PEFT (LoRA)
- Sentence Transformers
- Scikit-learn
- Pandas
- Matplotlib
- Gradio

---

## License

MIT License