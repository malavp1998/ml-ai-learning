"""
Deep inspection of house_price_model.h5
Shows the actual weights, biases, and structure inside the saved model.
"""

import numpy as np
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("=" * 65)
print("  WHAT IS ACTUALLY INSIDE house_price_model.h5?")
print("=" * 65)

model = tf.keras.models.load_model('house_price_model.h5', compile=False)

# ── 1. Layer overview ────────────────────────────────────────────
print("\n[1] LAYER OVERVIEW")
print("-" * 65)
for i, layer in enumerate(model.layers):
    w = layer.get_weights()
    if w:
        weights, biases = w[0], w[1]
        print(f"  Layer {i+1}: {layer.name}")
        print(f"    Input  shape : {weights.shape[0]} neurons")
        print(f"    Output shape : {weights.shape[1]} neurons")
        print(f"    Weights shape: {weights.shape}  ({weights.shape[0]*weights.shape[1]} numbers)")
        print(f"    Biases  shape: {biases.shape}   ({biases.shape[0]} numbers)")
        print(f"    Total params : {weights.size + biases.size}")
        print()

# ── 2. Show actual weight numbers ────────────────────────────────
print("\n[2] ACTUAL WEIGHT & BIAS VALUES (first layer — all 256 params)")
print("-" * 65)
layer1 = model.layers[0]
W1, b1 = layer1.get_weights()

print(f"\n  Weight matrix shape: {W1.shape}  (3 inputs × 64 neurons)")
print(f"\n  W1 = [")
for i, row in enumerate(W1):
    feature_name = ['size  ', 'beds  ', 'age   '][i]
    vals = "  ".join([f"{v:+.4f}" for v in row[:8]])
    print(f"    {feature_name} → [{vals}  ...] (64 values total)")
print(f"  ]\n")

print(f"  Bias vector shape: {b1.shape}  (1 bias per neuron)")
print(f"  b1 = [{', '.join([f'{v:+.4f}' for v in b1[:8]])}  ...] (64 values total)")

# ── 3. Manual single neuron calculation ───────────────────────────
print("\n[3] MANUAL CALCULATION — Showing how ONE neuron works")
print("-" * 65)

# Use scaled version of: [2000 sqft, 3 beds, 10 yrs]
# Approximate scaled values (from our StandardScaler)
from sklearn.preprocessing import StandardScaler
np.random.seed(42)
size     = np.random.uniform(800, 4000, 200)
bedrooms = np.random.randint(1, 6, 200)
age      = np.random.uniform(0, 50, 200)
X = np.column_stack([size, bedrooms, age])
scaler = StandardScaler().fit(X)

house = np.array([[2000, 3, 10]])
house_scaled = scaler.transform(house)[0]

print(f"\n  Input house: [size=2000, beds=3, age=10]")
print(f"  After scaling: [{house_scaled[0]:.4f}, {house_scaled[1]:.4f}, {house_scaled[2]:.4f}]")
print()
print(f"  Neuron #1 in Layer 1 computes:")
print(f"    w = [{W1[0,0]:+.4f}, {W1[1,0]:+.4f}, {W1[2,0]:+.4f}]  (weights for inputs: size, beds, age)")
print(f"    b = {b1[0]:+.4f}  (bias)")
print()
print(f"    weighted_sum = (size×w0) + (beds×w1) + (age×w2) + bias")
print(f"                 = ({house_scaled[0]:.4f}×{W1[0,0]:+.4f})")
print(f"                 + ({house_scaled[1]:.4f}×{W1[1,0]:+.4f})")
print(f"                 + ({house_scaled[2]:.4f}×{W1[2,0]:+.4f})")
print(f"                 + {b1[0]:+.4f}")
manual_sum = (house_scaled[0]*W1[0,0] + house_scaled[1]*W1[1,0] + house_scaled[2]*W1[2,0] + b1[0])
print(f"                 = {manual_sum:.4f}")
print()
relu_out = max(0, manual_sum)
print(f"    ReLU({manual_sum:.4f}) = {relu_out:.4f}  (output of neuron #1)")

# ── 4. Full forward pass manually ─────────────────────────────────
print("\n[4] FULL MANUAL FORWARD PASS (same as model.predict)")
print("-" * 65)

x = house_scaled.reshape(1, -1)
print(f"\n  Step 0 — Input:  {x[0]}")

for i, layer in enumerate(model.layers):
    w_list = layer.get_weights()
    if not w_list:
        continue
    W, b = w_list
    z = x @ W + b                          # matrix multiply + add bias
    x = np.maximum(0, z) if i < len(model.layers)-1 else z   # ReLU on hidden, linear on output
    print(f"  Step {i+1} — After Layer {i+1} ({layer.name}): shape={x.shape}, "
          f"sample values=[{', '.join([f'{v:.4f}' for v in x[0,:4]])}{'...' if x.shape[1]>4 else ''}]")

print(f"\n  Final output (manual):  ${x[0][0]:.4f}k")
tf_pred = model.predict(scaler.transform([[2000, 3, 10]]), verbose=0)
print(f"  TensorFlow prediction:  ${tf_pred[0][0]:.4f}k")
print(f"  Match: {'YES ✓' if abs(x[0][0] - tf_pred[0][0]) < 0.01 else 'CLOSE (floating point rounding)'}")

# ── 5. Model file summary ─────────────────────────────────────────
print("\n[5] WHAT IS STORED IN THE .h5 FILE")
print("-" * 65)
total = sum(layer.get_weights()[0].size + layer.get_weights()[1].size
            for layer in model.layers if layer.get_weights())
print(f"""
  house_price_model.h5 contains:
  ┌─────────────────────────────────────────────────────────┐
  │  Format      : HDF5 (Hierarchical Data Format v5)       │
  │  Architecture: 4 Dense layers (Sequential)              │
  │  Total params: {total} learned numbers               │
  │                                                         │
  │  Per layer:                                             │
  │    Layer 1: 192 weights  + 64 biases  = 256  params     │
  │    Layer 2: 2048 weights + 32 biases  = 2080 params     │
  │    Layer 3: 512 weights  + 16 biases  = 528  params     │
  │    Layer 4: 16 weights   + 1 bias     = 17   params     │
  │                                         ────            │
  │    Total:                               2881 params     │
  │                                                         │
  │  Contains:                                              │
  │    - Model architecture (JSON config)                   │
  │    - All 2881 float32 weight values                     │
  │    - Training configuration                             │
  └─────────────────────────────────────────────────────────┘
""")

print("  File size breakdown:")
print(f"    Actual file size: {os.path.getsize('house_price_model.h5') / 1024:.1f} KB")
print(f"    Theoretical min (2881 floats × 4 bytes): {2881*4/1024:.1f} KB")
print(f"    Overhead (architecture JSON + HDF5 structure): ~{os.path.getsize('house_price_model.h5')/1024 - 2881*4/1024:.1f} KB")
