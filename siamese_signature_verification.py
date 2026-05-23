"""
Siamese Neural Network for Signature Verification
Based on Bromley et al. (1993) — the original paper that invented Siamese Networks
specifically for offline signature verification.

Architecture: Twin networks sharing weights, trained with Contrastive Loss.
Task: Given two signatures, decide if they belong to the same person.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.decomposition import PCA
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.random.seed(42)
tf.random.set_seed(42)

# ── Hyperparameters ────────────────────────────────────────────────
FEATURE_DIM   = 128    # signature feature vector dimensions
NUM_PEOPLE    = 50     # registered users in the system
SAMPLES_EACH  = 20     # genuine signatures per person
EMBEDDING_DIM = 64     # compressed embedding size after base network
MARGIN        = 1.0    # contrastive loss margin
EPOCHS        = 60
BATCH_SIZE    = 32
NUM_PAIRS     = 4000   # training + test pairs

print("=" * 68)
print("   SIAMESE NEURAL NETWORK — SIGNATURE VERIFICATION SYSTEM")
print("   Based on Bromley et al. (1993)  |  Thesis Implementation")
print("=" * 68)

# ─────────────────────────────────────────────────────────────────
# STEP 1: Generate Synthetic Signature Data
# ─────────────────────────────────────────────────────────────────
def generate_signature_data():
    """
    Simulates signature feature vectors (128-dim).
    In a real system these come from preprocessing:
    → stroke direction, pen pressure, curvature, timing, aspect ratio, etc.

    Each person has a unique 'signature template'.
    Genuine samples  = template + small noise  (natural hand tremor)
    Skilled forgeries are handled by negative pairs from other people.
    """
    print("\n[Step 1] Generating synthetic signature dataset...")

    # Each person's unique signature fingerprint
    person_templates = np.random.randn(NUM_PEOPLE, FEATURE_DIM).astype(np.float32)

    signatures = []
    person_ids = []

    for pid in range(NUM_PEOPLE):
        for _ in range(SAMPLES_EACH):
            # Natural variation: ~15% noise relative to template norm
            noise = np.random.randn(FEATURE_DIM).astype(np.float32) * 0.15
            sig   = person_templates[pid] + noise
            signatures.append(sig)
            person_ids.append(pid)

    signatures = np.array(signatures, dtype=np.float32)
    person_ids = np.array(person_ids)

    print(f"   People in database  : {NUM_PEOPLE}")
    print(f"   Signatures per person: {SAMPLES_EACH}")
    print(f"   Total signatures    : {len(signatures)}")
    print(f"   Feature dimensions  : {FEATURE_DIM}")

    return signatures, person_ids, person_templates


def create_pairs(signatures, person_ids):
    """
    Create balanced genuine / forged pairs.
    Genuine (label=1): two signatures from the SAME person
    Forged  (label=0): signatures from TWO DIFFERENT people
    """
    print(f"\n[Step 2] Creating {NUM_PAIRS} signature pairs (50/50 balanced)...")

    person_index = {pid: np.where(person_ids == pid)[0]
                    for pid in range(NUM_PEOPLE)}

    pairs_A, pairs_B, pair_labels = [], [], []

    for _ in range(NUM_PAIRS // 2):
        # ── Genuine pair ──
        pid   = np.random.randint(NUM_PEOPLE)
        i1,i2 = np.random.choice(person_index[pid], 2, replace=False)
        pairs_A.append(signatures[i1])
        pairs_B.append(signatures[i2])
        pair_labels.append(1)

        # ── Forged pair ──
        p1, p2 = np.random.choice(NUM_PEOPLE, 2, replace=False)
        pairs_A.append(signatures[np.random.choice(person_index[p1])])
        pairs_B.append(signatures[np.random.choice(person_index[p2])])
        pair_labels.append(0)

    pairs_A     = np.array(pairs_A,     dtype=np.float32)
    pairs_B     = np.array(pairs_B,     dtype=np.float32)
    pair_labels = np.array(pair_labels, dtype=np.float32)

    print(f"   Genuine pairs: {int(pair_labels.sum())}")
    print(f"   Forged  pairs: {int((1-pair_labels).sum())}")

    return pairs_A, pairs_B, pair_labels


# ─────────────────────────────────────────────────────────────────
# STEP 2: Build the Siamese Architecture
# ─────────────────────────────────────────────────────────────────
def build_base_network():
    """
    The shared 'twin' network.
    Both input signatures pass through IDENTICAL weights.
    Learns to map raw features → compact embedding where:
      - Same person's signatures cluster together
      - Different people's signatures are pushed apart
    """
    inputs = keras.Input(shape=(FEATURE_DIM,), name='signature_features')

    x = keras.layers.Dense(256, activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(EMBEDDING_DIM, activation='relu')(x)

    # L2 normalize: project onto unit hypersphere
    # Ensures all embeddings have the same magnitude — only direction matters
    outputs = keras.layers.Lambda(
        lambda v: tf.math.l2_normalize(v, axis=1),
        name='l2_embedding'
    )(x)

    return keras.Model(inputs, outputs, name='base_network')


def build_siamese_model():
    """
    Full Siamese Network:
    Signature A ─┐
                 ├─► [Base Network] ─► Embedding A ─┐
    Signature B ─┘   (shared weights)                ├─► Distance ─► [0,∞)
                 └─► [Base Network] ─► Embedding B ─┘
    """
    base_network = build_base_network()

    input_a = keras.Input(shape=(FEATURE_DIM,), name='signature_A')
    input_b = keras.Input(shape=(FEATURE_DIM,), name='signature_B')

    # SAME network, SAME weights applied to both
    embedding_a = base_network(input_a)
    embedding_b = base_network(input_b)

    # Euclidean distance between embeddings
    distance = keras.layers.Lambda(
        lambda t: tf.sqrt(
            tf.maximum(tf.reduce_sum(tf.square(t[0] - t[1]), axis=1, keepdims=True), 1e-9)
        ),
        name='euclidean_distance'
    )([embedding_a, embedding_b])

    model = keras.Model(
        inputs=[input_a, input_b],
        outputs=distance,
        name='siamese_network'
    )
    return model, base_network


# ─────────────────────────────────────────────────────────────────
# STEP 3: Contrastive Loss
# ─────────────────────────────────────────────────────────────────
def contrastive_loss(y_true, y_pred):
    """
    Hadsell et al. (2006) Contrastive Loss:

      L = y * D²  +  (1-y) * max(margin - D, 0)²

      y=1 (genuine) : minimise D → pull same-person embeddings together
      y=0 (forged)  : maximise D (up to margin) → push apart

    The margin prevents the network from pushing negatives to infinity
    and ensures a clear separation boundary.
    """
    D    = y_pred
    loss = (y_true * tf.square(D)
            + (1.0 - y_true) * tf.square(tf.maximum(MARGIN - D, 0.0)))
    return tf.reduce_mean(loss)


# ─────────────────────────────────────────────────────────────────
# STEP 4: Train
# ─────────────────────────────────────────────────────────────────
signatures, person_ids, templates = generate_signature_data()
pairs_A, pairs_B, labels           = create_pairs(signatures, person_ids)

# 80 / 20 split
split = int(0.8 * len(labels))
tr_A, te_A = pairs_A[:split], pairs_A[split:]
tr_B, te_B = pairs_B[:split], pairs_B[split:]
tr_y, te_y = labels[:split],  labels[split:]
print(f"\n   Training pairs: {len(tr_y)} | Test pairs: {len(te_y)}")

print("\n[Step 3] Building Siamese Network...")
model, base_net = build_siamese_model()

print("\n   BASE NETWORK (identical copy used for both inputs):")
base_net.summary()
print("\n   FULL SIAMESE MODEL:")
model.summary()

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss=contrastive_loss
)

print("\n[Step 4] Training with Contrastive Loss...")
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    ),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )
]

history = model.fit(
    [tr_A, tr_B], tr_y,
    validation_split=0.15,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)


# ─────────────────────────────────────────────────────────────────
# STEP 5: Evaluate
# ─────────────────────────────────────────────────────────────────
print("\n[Step 5] Evaluating signature verification performance...")
distances = model.predict([te_A, te_B], verbose=0).flatten()

# ROC curve — negate distances because lower distance = more genuine
fpr, tpr, thresholds = roc_curve(te_y, -distances)
roc_auc = auc(fpr, tpr)

# Equal Error Rate: where FAR = FRR
fnr     = 1.0 - tpr
eer_idx = np.argmin(np.abs(fnr - fpr))
eer     = (fpr[eer_idx] + fnr[eer_idx]) / 2.0
opt_thr = -thresholds[eer_idx]

# Accuracy, FAR, FRR at EER threshold
preds = (distances < opt_thr).astype(int)
acc   = np.mean(preds == te_y.astype(int))
cm    = confusion_matrix(te_y.astype(int), preds)
FAR   = cm[0,1] / (cm[0,0] + cm[0,1]) if (cm[0,0]+cm[0,1]) > 0 else 0
FRR   = cm[1,0] / (cm[1,0] + cm[1,1]) if (cm[1,0]+cm[1,1]) > 0 else 0

print(f"\n   ┌─────────────────────────────────────────────┐")
print(f"   │  VERIFICATION RESULTS                       │")
print(f"   │                                             │")
print(f"   │  Accuracy             : {acc*100:>6.2f}%            │")
print(f"   │  Equal Error Rate     : {eer*100:>6.2f}%            │")
print(f"   │  AUC-ROC              : {roc_auc:>7.4f}           │")
print(f"   │  FAR (False Accept)   : {FAR*100:>6.2f}%            │")
print(f"   │  FRR (False Reject)   : {FRR*100:>6.2f}%            │")
print(f"   │  Optimal Threshold    : {opt_thr:>7.4f}           │")
print(f"   └─────────────────────────────────────────────┘")


# ─────────────────────────────────────────────────────────────────
# STEP 6: Live verification demo
# ─────────────────────────────────────────────────────────────────
print("\n[Step 6] Live Verification Demos...")
print("-" * 68)

test_cases = [
    ("Genuine pair",   0, 0,  "Same person  → should ACCEPT"),
    ("Forged  pair",   0, 1,  "Diff person  → should REJECT"),
    ("Forged  pair",   5, 12, "Diff person  → should REJECT"),
    ("Genuine pair",   7, 7,  "Same person  → should ACCEPT"),
]

for label, p1, p2, note in test_cases:
    i1 = np.where(person_ids == p1)[0][0]
    i2 = np.where(person_ids == p2)[0][0 if p1 != p2 else 1]
    d  = model.predict([signatures[i1:i1+1], signatures[i2:i2+1]], verbose=0)[0][0]
    verdict = "ACCEPT ✓" if d < opt_thr else "REJECT ✗"
    print(f"  {label:15s}  Person {p1:2d} vs Person {p2:2d}  │  "
          f"Dist={d:.4f}  │  {verdict}  ({note})")

print(f"\n  Threshold = {opt_thr:.4f}  (below → same person, above → forged)")


# ─────────────────────────────────────────────────────────────────
# STEP 7: Visualize results
# ─────────────────────────────────────────────────────────────────
print("\n[Step 7] Generating visualizations...")

fig = plt.figure(figsize=(18, 12))
fig.suptitle("Siamese Neural Network — Signature Verification System",
             fontsize=15, fontweight='bold', y=0.98)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# ── 1. Training loss ──────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 0])
ax.plot(history.history['loss'],     color='steelblue', lw=2, label='Train Loss')
ax.plot(history.history['val_loss'], color='tomato',    lw=2, linestyle='--', label='Val Loss')
ax.set_title('Contrastive Loss\nDuring Training', fontweight='bold')
ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
ax.legend(); ax.grid(alpha=0.3)

# ── 2. Distance distribution ──────────────────────────────────────
ax = fig.add_subplot(gs[0, 1])
gen_d = distances[te_y == 1]
for_d = distances[te_y == 0]
ax.hist(gen_d, bins=35, alpha=0.7, color='seagreen',  label=f'Genuine (n={len(gen_d)})', density=True)
ax.hist(for_d, bins=35, alpha=0.7, color='crimson',   label=f'Forged  (n={len(for_d)})', density=True)
ax.axvline(opt_thr, color='black', lw=2, linestyle='--', label=f'Threshold={opt_thr:.3f}')
ax.set_title('Distance Distribution\nGenuine vs Forged', fontweight='bold')
ax.set_xlabel('Euclidean Distance'); ax.set_ylabel('Density')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── 3. ROC curve ──────────────────────────────────────────────────
ax = fig.add_subplot(gs[0, 2])
ax.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC  (AUC={roc_auc:.3f})')
ax.plot([0,1], [0,1], 'k--', alpha=0.3, label='Random classifier')
ax.scatter(fpr[eer_idx], tpr[eer_idx], s=120, color='red', zorder=5, label=f'EER={eer*100:.1f}%')
ax.set_title('ROC Curve', fontweight='bold')
ax.set_xlabel('FAR (False Acceptance Rate)')
ax.set_ylabel('TAR (True Acceptance Rate)')
ax.legend(fontsize=9); ax.grid(alpha=0.3)

# ── 4. Embedding space (PCA 2D) ───────────────────────────────────
ax = fig.add_subplot(gs[1, 0])
sample_sigs = signatures[:100]
sample_pids = person_ids[:100]
embeddings  = base_net.predict(sample_sigs, verbose=0)
emb_2d      = PCA(n_components=2).fit_transform(embeddings)
colors = plt.cm.tab10(np.linspace(0, 1, 5))
for i, pid in enumerate(range(5)):
    mask = sample_pids == pid
    ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
               color=colors[i], s=60, label=f'Person {pid}', alpha=0.8)
ax.set_title('Embedding Space\n(PCA 2D, first 5 people)', fontweight='bold')
ax.set_xlabel('PC 1'); ax.set_ylabel('PC 2')
ax.legend(fontsize=8); ax.grid(alpha=0.3)

# ── 5. Confusion matrix ───────────────────────────────────────────
ax = fig.add_subplot(gs[1, 1])
im = ax.imshow(cm, cmap='Blues')
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(['Predicted\nForged', 'Predicted\nGenuine'], fontsize=9)
ax.set_yticklabels(['Actual\nForged', 'Actual\nGenuine'], fontsize=9)
ax.set_title('Confusion Matrix', fontweight='bold')
labels_cm = [['TN\n(Correct Reject)', 'FP\n(False Accept)'],
             ['FN\n(False Reject)',   'TP\n(Correct Accept)']]
for i in range(2):
    for j in range(2):
        ax.text(j, i, f"{cm[i,j]}\n{labels_cm[i][j]}",
                ha='center', va='center', fontsize=8,
                color='white' if cm[i,j] > cm.max()/2 else 'black')
plt.colorbar(im, ax=ax)

# ── 6. FAR / FRR tradeoff ─────────────────────────────────────────
ax = fig.add_subplot(gs[1, 2])
t_range = np.linspace(distances.min(), distances.max(), 200)
fars_, frrs_ = [], []
for t in t_range:
    p_ = (distances < t).astype(int)
    tp_ = np.sum((p_==1)&(te_y==1)); fp_ = np.sum((p_==1)&(te_y==0))
    fn_ = np.sum((p_==0)&(te_y==1)); tn_ = np.sum((p_==0)&(te_y==0))
    fars_.append(fp_/(fp_+tn_) if (fp_+tn_)>0 else 0)
    frrs_.append(fn_/(fn_+tp_) if (fn_+tp_)>0 else 0)
ax.plot(t_range, fars_, color='red',    lw=2, label='FAR')
ax.plot(t_range, frrs_, color='royalblue', lw=2, label='FRR')
ax.axvline(opt_thr, color='black', lw=1.5, linestyle='--', label=f'EER @ {opt_thr:.3f}')
ax.fill_between(t_range, fars_, frrs_, alpha=0.08, color='purple')
ax.set_title('FAR / FRR vs Threshold\n(EER = crossover point)', fontweight='bold')
ax.set_xlabel('Decision Threshold'); ax.set_ylabel('Error Rate')
ax.legend(); ax.grid(alpha=0.3)

plt.savefig('siamese_results.png', dpi=120, bbox_inches='tight')
print("   Saved → siamese_results.png")

# Save
model.save('siamese_model.h5')
base_net.save('base_network.h5')
print("   Saved → siamese_model.h5 | base_network.h5")

print("\n" + "=" * 68)
print("   TRAINING COMPLETE")
print("=" * 68)
print(f"\n   Accuracy  : {acc*100:.2f}%  |  EER: {eer*100:.2f}%  |  AUC: {roc_auc:.4f}")
print(f"   FAR       : {FAR*100:.2f}%   |  FRR: {FRR*100:.2f}%")
