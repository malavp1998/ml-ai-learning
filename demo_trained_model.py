"""
Demo: Load and use the pre-trained house price model
Shows predictions, analysis, and visualizations
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Suppress TF info messages
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("  LOADING TRAINED MODEL & RUNNING PREDICTIONS")
print("=" * 60)

# ── Load trained model ──────────────────────────────────────────
print("\n[1] Loading saved model from house_price_model.h5 ...")
model = tf.keras.models.load_model('house_price_model.h5', compile=False)
print("    Model loaded successfully!")
print(f"    Layers  : {len(model.layers)}")
print(f"    Params  : {model.count_params():,}")

# ── Re-create the scaler (same seed = same data = same scaler) ──
def generate_house_data(num_samples=200):
    size     = np.random.uniform(800, 4000, num_samples)
    bedrooms = np.random.randint(1, 6, num_samples)
    age      = np.random.uniform(0, 50, num_samples)
    price    = (100 + size * 0.15 + bedrooms * 50 - age * 1.5
                + np.random.normal(0, 30, num_samples))
    price    = np.maximum(price, 50)
    return np.column_stack([size, bedrooms, age]), price

X, y = generate_house_data(200)
scaler = StandardScaler()
scaler.fit(X)          # fit on full dataset to reproduce training scaler

# ── Custom house predictions ─────────────────────────────────────
print("\n[2] Predicting prices for custom house configurations ...")

custom_houses = [
    {"name": "Starter Home",   "size": 900,  "beds": 2, "age": 25},
    {"name": "Family Home",    "size": 1800, "beds": 3, "age": 10},
    {"name": "Luxury Villa",   "size": 3500, "beds": 5, "age": 2},
    {"name": "Old Bungalow",   "size": 1200, "beds": 3, "age": 45},
    {"name": "New Townhouse",  "size": 2200, "beds": 4, "age": 1},
    {"name": "Mid-size Condo", "size": 1500, "beds": 2, "age": 15},
]

features = np.array([[h["size"], h["beds"], h["age"]] for h in custom_houses])
features_scaled = scaler.transform(features)
predictions = model.predict(features_scaled, verbose=0).flatten()

print(f"\n{'House Type':<18} {'Size':>8} {'Beds':>6} {'Age':>5} {'Predicted Price':>17}")
print("-" * 58)
for h, pred in zip(custom_houses, predictions):
    print(f"{h['name']:<18} {h['size']:>7} sqft  {h['beds']:>2}bd  "
          f"{h['age']:>3}yr  ${pred:>10,.0f}k  (${pred*1000:>10,.0f})")

# ── Size sensitivity analysis ────────────────────────────────────
print("\n[3] Size sensitivity: how price changes with square footage ...")
sizes      = np.linspace(800, 4000, 100)
size_feats = np.column_stack([sizes, np.full(100, 3), np.full(100, 10)])
size_preds = model.predict(scaler.transform(size_feats), verbose=0).flatten()

# ── Age sensitivity analysis ─────────────────────────────────────
ages      = np.linspace(0, 50, 100)
age_feats = np.column_stack([np.full(100, 2000), np.full(100, 3), ages])
age_preds = model.predict(scaler.transform(age_feats), verbose=0).flatten()

# ── Bedroom comparison ───────────────────────────────────────────
bed_labels = [1, 2, 3, 4, 5]
bed_feats  = np.array([[2000, b, 10] for b in bed_labels])
bed_preds  = model.predict(scaler.transform(bed_feats), verbose=0).flatten()

# ── Heatmap: size vs age ─────────────────────────────────────────
s_range = np.linspace(800, 4000, 30)
a_range = np.linspace(0, 50, 30)
S, A    = np.meshgrid(s_range, a_range)
heat_feats = np.column_stack([S.ravel(), np.full(900, 3), A.ravel()])
heat_preds = model.predict(scaler.transform(heat_feats), verbose=0).flatten().reshape(30, 30)

# ── Plot everything ──────────────────────────────────────────────
print("\n[4] Generating comprehensive visualization ...")

fig = plt.figure(figsize=(16, 12))
fig.suptitle("House Price Prediction – Trained Neural Network Analysis",
             fontsize=15, fontweight='bold', y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

# -- Bar chart: custom house predictions
ax1 = fig.add_subplot(gs[0, :2])
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(custom_houses)))
bars = ax1.barh([h["name"] for h in custom_houses], predictions, color=colors)
ax1.set_xlabel("Predicted Price ($k)", fontsize=11)
ax1.set_title("Custom House Predictions", fontsize=12, fontweight='bold')
ax1.bar_label(bars, labels=[f"${p:,.0f}k" for p in predictions],
              padding=4, fontsize=9)
ax1.set_xlim(0, max(predictions) * 1.2)
ax1.grid(axis='x', alpha=0.3)

# -- Bedroom bar chart
ax2 = fig.add_subplot(gs[0, 2])
bar_colors = plt.cm.plasma(np.linspace(0.2, 0.8, 5))
b = ax2.bar([str(b) for b in bed_labels], bed_preds, color=bar_colors)
ax2.set_xlabel("Number of Bedrooms", fontsize=10)
ax2.set_ylabel("Predicted Price ($k)", fontsize=10)
ax2.set_title("Price vs Bedrooms\n(2000 sqft, 10yr old)", fontsize=11, fontweight='bold')
ax2.bar_label(b, labels=[f"${p:,.0f}k" for p in bed_preds], padding=3, fontsize=8)
ax2.set_ylim(0, max(bed_preds) * 1.15)
ax2.grid(axis='y', alpha=0.3)

# -- Size sensitivity line
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(sizes, size_preds, color='steelblue', linewidth=2.5)
ax3.fill_between(sizes, size_preds, alpha=0.15, color='steelblue')
ax3.set_xlabel("House Size (sq ft)", fontsize=10)
ax3.set_ylabel("Predicted Price ($k)", fontsize=10)
ax3.set_title("Price vs House Size\n(3 beds, 10yr)", fontsize=11, fontweight='bold')
ax3.grid(alpha=0.3)

# -- Age sensitivity line
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(ages, age_preds, color='tomato', linewidth=2.5)
ax4.fill_between(ages, age_preds, alpha=0.15, color='tomato')
ax4.set_xlabel("House Age (years)", fontsize=10)
ax4.set_ylabel("Predicted Price ($k)", fontsize=10)
ax4.set_title("Price vs House Age\n(2000 sqft, 3 beds)", fontsize=11, fontweight='bold')
ax4.grid(alpha=0.3)

# -- Heatmap: size vs age
ax5 = fig.add_subplot(gs[1, 2])
hm = ax5.contourf(S, A, heat_preds, levels=20, cmap='RdYlGn')
fig.colorbar(hm, ax=ax5, label='Price ($k)')
ax5.set_xlabel("House Size (sq ft)", fontsize=10)
ax5.set_ylabel("House Age (years)", fontsize=10)
ax5.set_title("Price Heatmap\n(3 bedrooms)", fontsize=11, fontweight='bold')

out_path = '/Users/umamalav/Documents/Projects/ML-AI/model_demo_output.png'
plt.savefig(out_path, dpi=120, bbox_inches='tight')
print(f"    Saved → model_demo_output.png")
plt.close()

# ── Summary ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DEMO COMPLETE")
print("=" * 60)
print(f"\n  Lowest prediction : ${min(predictions):,.0f}k  ({custom_houses[np.argmin(predictions)]['name']})")
print(f"  Highest prediction: ${max(predictions):,.0f}k  ({custom_houses[np.argmax(predictions)]['name']})")
print(f"  Price lift per +1 bed    : ~${(bed_preds[-1]-bed_preds[0])/4:,.0f}k")
print(f"  Price drop per +10 yr age: ~${(age_preds[0]-age_preds[-1])/5:,.0f}k")
print(f"  Price lift per +1000 sqft: ~${(size_preds[-1]-size_preds[0])/3.2:,.0f}k")
print("\n  Visualization saved to model_demo_output.png")
