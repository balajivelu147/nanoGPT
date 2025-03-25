"""
Prepare a custom dataset for character-level language modeling.
This reads a local input file, maps characters to integers,
and saves train.bin, val.bin, and meta.pkl for training.
"""

import os
import pickle
import numpy as np

# ===== CONFIG: Set Your Input File Path =====
# Replace with your dataset's path (absolute or relative)
input_file_path = os.path.join(os.path.dirname(__file__), 'input.txt')  # Your local file

# ===== Load Your Data =====
if not os.path.exists(input_file_path):
    raise FileNotFoundError(f"Input file not found at {input_file_path}. Please check the path.")

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = f.read()
print(f"✅ Loaded dataset. Length in characters: {len(data):,}")

# ===== Create Character-Level Vocabulary =====
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"✅ Unique characters ({vocab_size}): {''.join(chars)}")

# ===== Create Encoding and Decoding Maps =====
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# ===== Train/Validation Split =====
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

# ===== Encode to Integer IDs =====
train_ids = np.array(encode(train_data), dtype=np.uint16)
val_ids = np.array(encode(val_data), dtype=np.uint16)

print(f"✅ Train tokens: {len(train_ids):,}, Validation tokens: {len(val_ids):,}")

# ===== Save Encoded Data =====
train_bin_path = os.path.join(os.path.dirname(__file__), 'train.bin')
val_bin_path = os.path.join(os.path.dirname(__file__), 'val.bin')

train_ids.tofile(train_bin_path)
val_ids.tofile(val_bin_path)
print(f"💾 Saved train.bin and val.bin")

# ===== Save Vocabulary and Metadata =====
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

meta_path = os.path.join(os.path.dirname(__file__), 'meta.pkl')
with open(meta_path, 'wb') as f:
    pickle.dump(meta, f)
print(f"💾 Saved meta.pkl")

print("\n🎉 Dataset preparation complete. Ready for training NanoGPT!")
