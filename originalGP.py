import os
import pickle
import torch
from model import GPTConfig, GPT
from tqdm import tqdm

# ---------------- CONFIG -------------------
out_dir = 'out-shakespeare-char'    # Folder with ckpt.pt and meta.pkl
input_file = 'data/shakespeare_char/input.txt'  # Evaluation input
device = 'cpu'                      # 'cpu' or 'cuda'
block_size = 64                     # Must match training block size
max_eval_chars = 100000               # Evaluate first N chars (None for full)
# ------------------------------------------

# Load model
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device)
model_args = checkpoint['model_args']
model = GPT(GPTConfig(**model_args))
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()

# Load tokenizer
meta_path = os.path.join(out_dir, 'meta.pkl')
with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

# Fallback logic
fallback_char = ' ' if ' ' in stoi else list(stoi.keys())[0]
encode = lambda s: [stoi.get(c, stoi[fallback_char]) for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# Load input
with open(input_file, 'r', encoding='utf-8') as f:
    data = f.read()
if max_eval_chars is not None:
    data = data[:max_eval_chars]
encoded_data = encode(data)

# Character grouping logic (CAPITAL letters)
group1 = set(['A', 'B', 'C', 'D', 'E'])
group2 = set(['F', 'G', 'H', 'I', 'J'])


def in_same_group(pred_char, actual_char):
    if pred_char == actual_char:
        return True  # Exact match is always valid
    elif pred_char in group1 and actual_char in group1:
        return True
    elif pred_char in group2 and actual_char in group2:
        return True
    return False

# Evaluate
correct = 0
total = 0

print(f"🔍 Grouped accuracy evaluation on {len(encoded_data)} characters...")
for i in tqdm(range(1, len(encoded_data))):
    start_idx = max(0, i - block_size)
    context = encoded_data[start_idx:i]
    context_tensor = torch.tensor(context, dtype=torch.long, device=device)[None, ...]

    with torch.no_grad():
        logits, _ = model(context_tensor)
        next_char_logits = logits[0, -1]
        predicted_id = torch.argmax(next_char_logits).item()

    actual_id = encoded_data[i]
    pred_char = itos[predicted_id]
    actual_char = itos[actual_id]

    if in_same_group(pred_char, actual_char):
        correct += 1
    total += 1

# Results
accuracy = correct / total if total > 0 else 0
print(f"\n✅ Grouped prediction accuracy: {accuracy * 100:.2f}% ({correct} correct out of {total})")
