from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "microsoft/Phi-4-mini-instruct"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    trust_remote_code=True
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    trust_remote_code=True,
    torch_dtype="auto",
    device_map="cpu"
)

print("Success!")