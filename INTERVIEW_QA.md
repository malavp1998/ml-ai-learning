# TensorFlow Interview Q&A — From Basics to Advanced

> All answers connect back to the House Price Prediction project you built.
> Questions are ordered from beginner → intermediate → advanced.

---

## Table of Contents

1. [Python & Data Fundamentals](#1-python--data-fundamentals)
2. [Machine Learning Core Concepts](#2-machine-learning-core-concepts)
3. [Neural Networks — Theory](#3-neural-networks--theory)
4. [TensorFlow & Keras](#4-tensorflow--keras)
5. [Project Deep Dive](#5-project-deep-dive)
6. [Model Evaluation & Improvement](#6-model-evaluation--improvement)
7. [Advanced Topics](#7-advanced-topics)
8. [Scenario / Problem-Solving Questions](#8-scenario--problem-solving-questions)

---

## 1. Python & Data Fundamentals

---

### Q1. What is NumPy and why do we use it in ML projects?

**Answer:**

NumPy (Numerical Python) is a library for working with arrays and matrices of numbers. It is the backbone of almost every ML project.

**Why we need it:**
- Python lists are slow for math operations
- NumPy arrays use contiguous memory and C-level speed
- All ML frameworks (TensorFlow, scikit-learn) accept NumPy arrays as input

**In our project:**
```python
import numpy as np

# We generated house features as NumPy arrays
size     = np.random.uniform(800, 4000, 200)   # 200 random sizes
bedrooms = np.random.randint(1, 6, 200)         # 200 random bedroom counts
age      = np.random.uniform(0, 50, 200)        # 200 random ages

# Stack three 1D arrays into one 2D array (200 rows × 3 columns)
X = np.column_stack([size, bedrooms, age])
print(X.shape)   # → (200, 3)
```

**Key NumPy operations to know:**
| Operation | Code | Result |
|-----------|------|--------|
| Create array | `np.array([1,2,3])` | `[1 2 3]` |
| Random numbers | `np.random.uniform(0, 1, 5)` | 5 random floats |
| Shape of array | `X.shape` | `(200, 3)` |
| Stack arrays | `np.column_stack([a, b])` | 2D matrix |
| Math operations | `X * 2`, `X + 5` | Element-wise ops |

---

### Q2. What is `train_test_split` and why do we split data?

**Answer:**

`train_test_split` from scikit-learn divides your dataset into two parts:

- **Training set**: Data the model *learns* from
- **Test set**: Data the model has *never seen* — used to evaluate real-world performance

**Why this matters:**

Imagine you give a student the exam questions beforehand. They score 100% — but do they really understand the subject? No.

Same logic applies here: if you test the model on data it trained on, it looks great but fails in production. The test set simulates real unseen data.

**In our project:**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% for testing (40 houses)
    random_state=42   # seed for reproducibility
)

print(X_train.shape)  # (160, 3) — 160 houses for training
print(X_test.shape)   # (40, 3)  — 40 houses for testing
```

**Common splits:**
- 80/20 (most common, our project)
- 70/30 (small datasets)
- 60/20/20 (train / validation / test)

---

### Q3. What is StandardScaler and why is normalisation important?

**Answer:**

`StandardScaler` transforms each feature to have **mean = 0** and **standard deviation = 1**.

**Formula:**
```
scaled_value = (original_value - mean) / std_deviation
```

**Why neural networks need this:**

Our three features have very different scales:
- House size: 800 – 4000 (range of 3200)
- Bedrooms: 1 – 5 (range of 4)
- Age: 0 – 50 (range of 50)

Without scaling, the network treats "size" as far more important just because its numbers are bigger. The gradient updates become uneven and training is unstable.

**Analogy:** Comparing apples and elephants. If you measure both in grams, the elephant always wins even if you only care about colour.

**In our project:**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Fit ONLY on training data — then transform both sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)     # NO fit here — critical!
```

**Why NOT fit on test data?**
Fitting on test data would leak information about the test set into your scaler — this is called **data leakage**. Always fit on training data only.

**After scaling (approximate values):**
```
Size 2000 sqft  → ~0.05  (near the mean)
Size 800 sqft   → ~-1.2  (below mean)
Size 4000 sqft  → ~1.3   (above mean)
```

---

## 2. Machine Learning Core Concepts

---

### Q4. What is Machine Learning? How is it different from traditional programming?

**Answer:**

**Traditional Programming:**
```
Rules + Data → Output
```
You write explicit if/else rules. The computer follows your instructions exactly.

**Machine Learning:**
```
Data + Output → Rules (learned automatically)
```
You show the algorithm examples. It figures out the rules on its own.

**Concrete example from our project:**

Traditional (you write rules):
```python
# You manually define the formula
def predict_price(size, beds, age):
    return 100 + (size * 0.15) + (beds * 50) - (age * 1.5)
```

Machine Learning (model learns the formula):
```python
# You just show examples — model discovers the formula
model.fit(X_train, y_train)   # model learns the relationship
model.predict([[2000, 3, 10]])  # $428k — learned by itself!
```

**Types of ML:**
| Type | Description | Example |
|------|-------------|---------|
| **Supervised** | Learn from labeled data | Our project (price is the label) |
| **Unsupervised** | Find patterns, no labels | Customer clustering |
| **Reinforcement** | Learn by rewards/penalties | Game-playing AI |

Our project is **supervised regression** (predicting a continuous number).

---

### Q5. What is the difference between Regression and Classification?

**Answer:**

| | Regression | Classification |
|--|-----------|----------------|
| **Output** | Continuous number | Discrete category |
| **Example** | House price = $428,000 | House is "affordable" or "luxury" |
| **Loss function** | MSE, MAE, RMSE | Cross-entropy, accuracy |
| **Output activation** | None (linear) | Sigmoid (binary), Softmax (multi) |

**In our project:**
We predict a price (a continuous number), so this is **regression**.

If we instead predicted "cheap / mid / expensive", that would be **classification**.

**Code difference:**
```python
# Regression (our project)
model.add(keras.layers.Dense(1))           # 1 output neuron, no activation
model.compile(loss='mse')                   # Mean Squared Error

# Classification (if we categorised)
model.add(keras.layers.Dense(3, activation='softmax'))  # 3 classes
model.compile(loss='categorical_crossentropy')
```

---

### Q6. What is overfitting and underfitting?

**Answer:**

These are the two core failure modes of any ML model.

**Overfitting:** Model memorises training data but fails on new data.
- Training loss: very low
- Test loss: much higher
- Like a student who memorises past exams but can't solve new problems

**Underfitting:** Model is too simple to learn the patterns.
- Both training and test loss are high
- Like a student who barely studied

**Just right (generalisation):** Model learns the true underlying pattern.
- Training and test loss are both low and close to each other

```
Training Loss:   ████░░   (low)
Validation Loss: ████░░   (similarly low)  ← GOOD — generalises
```

```
Training Loss:   ██░░░░   (very low)
Validation Loss: ████████ (very high)      ← BAD — overfit
```

**How to fix overfitting:**
```python
# Add Dropout layers
keras.layers.Dropout(0.2)   # randomly turns off 20% of neurons

# Reduce model complexity (fewer neurons)

# Get more training data

# Add regularisation
keras.layers.Dense(64, kernel_regularizer='l2')
```

**How to fix underfitting:**
- Add more layers or neurons
- Train for more epochs
- Use a more powerful model

---

## 3. Neural Networks — Theory

---

### Q7. What is a neuron and what does it do?

**Answer:**

A neuron is the basic unit of a neural network. It takes multiple inputs, applies a weighted sum, adds a bias, then passes the result through an activation function.

**Mathematical formula:**
```
output = activation( (w1×x1) + (w2×x2) + (w3×x3) + bias )
```

**Visual:**
```
x1 ──w1──┐
x2 ──w2──┤──[ Σ + bias ]──[ activation ]──► output
x3 ──w3──┘
```

**In our project — first layer neuron example:**
```
inputs:  [size=2000, bedrooms=3, age=10]  (after scaling)

neuron computes:
  weighted_sum = (0.3 × 2000) + (-0.1 × 3) + (-0.05 × 10) + 0.5
               = 600 - 0.3 - 0.5 + 0.5
               = 599.7

  output = ReLU(599.7) = 599.7   (since 599.7 > 0)
```

The **weights** (w1, w2, w3) and **bias** are what the model *learns* during training. Initially random, they get adjusted every epoch to make better predictions.

---

### Q8. What is an activation function? Why do we use ReLU?

**Answer:**

An activation function adds **non-linearity** to the network. Without it, stacking layers of neurons is mathematically equivalent to a single layer — the network can only learn linear relationships.

**Common activation functions:**

| Function | Formula | Output Range | When to use |
|----------|---------|-------------|-------------|
| **ReLU** | `max(0, x)` | 0 to ∞ | Hidden layers (default choice) |
| **Sigmoid** | `1/(1+e^-x)` | 0 to 1 | Binary classification output |
| **Softmax** | `e^xi / Σe^x` | 0 to 1 (sums to 1) | Multi-class classification output |
| **Linear** | `x` | -∞ to ∞ | Regression output (our project) |
| **Tanh** | `(e^x - e^-x)/(e^x + e^-x)` | -1 to 1 | Sometimes hidden layers |

**Why ReLU is the default for hidden layers:**
- Simple and fast to compute
- Doesn't suffer from the "vanishing gradient" problem
- Works well in practice for most networks

**In our project:**
```python
keras.layers.Dense(64, activation='relu')  # Hidden layer — ReLU
keras.layers.Dense(1)                      # Output layer — linear (no activation)
```

The output layer has NO activation because we want to predict any positive price value. If we used sigmoid, it would cap at 1.0.

**Visual of ReLU:**
```
output
  |      /
  |     /
  |    /
  |   /
  |  /
──|──────────── input
  0
```
Negative input → 0. Positive input → passes through unchanged.

---

### Q9. What is backpropagation? Explain it simply.

**Answer:**

Backpropagation is the algorithm that teaches the network to improve. It works by calculating how much each weight contributed to the prediction error, then adjusting weights to reduce that error.

**Step by step:**

**Step 1 — Forward Pass:** Feed input through the network, get a prediction.
```
Input [2000 sqft, 3 beds, 10yr] → Model → Predicted: $500k
```

**Step 2 — Calculate Error (Loss):**
```
Actual price: $428k
Predicted:    $500k
Error (Loss): (500 - 428)² = 5184  ← MSE
```

**Step 3 — Backward Pass:** Calculate how much each weight contributed to the error.
- Uses calculus (chain rule of differentiation)
- Computes the "gradient" — the slope of the error relative to each weight

**Step 4 — Update Weights:**
```
new_weight = old_weight - (learning_rate × gradient)
```
If the gradient is positive, reduce the weight. If negative, increase it.

**Analogy:** You're hiking down a mountain blindfolded. At each step, you feel the ground to find which direction is downhill, and take a small step that way. Eventually you reach the valley (minimum loss).

**In our project:**
```python
# This triggers backpropagation automatically every epoch
model.fit(X_train_scaled, y_train, epochs=30)
# TensorFlow calculates all gradients using tf.GradientTape internally
```

---

### Q10. What is a loss function? What is MSE?

**Answer:**

A loss function measures how wrong the model's predictions are. The goal of training is to **minimise** this value.

**MSE — Mean Squared Error:**
```
MSE = (1/n) × Σ (predicted - actual)²
```

**Why squared?**
- Makes all errors positive (no cancellation between over and under)
- Penalises large errors more heavily than small ones
- Mathematically convenient for derivatives

**Example from our project:**
```
House 1: Predicted $450k, Actual $428k → error = (450-428)² = 484
House 2: Predicted $380k, Actual $325k → error = (380-325)² = 3025
House 3: Predicted $620k, Actual $619k → error = (620-619)² = 1

MSE = (484 + 3025 + 1) / 3 = 1170
```

**Other loss functions:**
| Function | Formula | Best when |
|----------|---------|-----------|
| MSE | mean((y_pred - y_actual)²) | Regression, few outliers |
| MAE | mean(|y_pred - y_actual|) | Regression, many outliers |
| RMSE | sqrt(MSE) | Same scale as target |
| Cross-entropy | -Σ y·log(p) | Classification |

**In our project:**
```python
model.compile(loss='mse', metrics=['mae'])
# loss='mse'  → used to UPDATE weights (drives learning)
# metrics=['mae'] → used to MONITOR progress (human-readable)
```

---

## 4. TensorFlow & Keras

---

### Q11. What is TensorFlow? What is Keras?

**Answer:**

**TensorFlow** is Google's open-source ML framework. It handles:
- Efficient numerical computation (using tensors)
- Automatic differentiation (gradients for backpropagation)
- GPU/TPU acceleration
- Model deployment (TensorFlow Serving, TensorFlow Lite)

**Keras** is a high-level API built *on top of* TensorFlow. It simplifies building models with a clean, readable interface. Since TensorFlow 2.0, Keras is included as `tf.keras`.

**Analogy:**
- TensorFlow = Car engine (powerful but complex)
- Keras = Steering wheel + pedals (easy interface to control the engine)

**In our project:**
```python
import tensorflow as tf
from tensorflow import keras          # high-level API

# Keras makes model building readable:
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1)
])
```

Without Keras, the same model in raw TensorFlow would require 50+ lines of manual matrix math.

---

### Q12. What is `keras.Sequential`? When would you NOT use it?

**Answer:**

`Sequential` is the simplest way to build a model — layers are stacked one after another, each layer's output becoming the next layer's input.

```python
model = keras.Sequential([
    keras.layers.Input(shape=(3,)),   # 3 input features
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])
```

**Data flows linearly:**
```
Input (3) → Dense(64) → Dense(32) → Dense(16) → Output (1)
```

**When Sequential is NOT enough:**

1. **Multiple inputs** (e.g., image + text → prediction)
2. **Multiple outputs** (e.g., predict price AND category)
3. **Residual/skip connections** (used in ResNet)
4. **Branching architectures**

For those cases you use the **Functional API**:
```python
inputs = keras.Input(shape=(3,))
x = keras.layers.Dense(64, activation='relu')(inputs)
x = keras.layers.Dense(32, activation='relu')(x)
outputs = keras.layers.Dense(1)(x)

model = keras.Model(inputs=inputs, outputs=outputs)
```

Our project uses Sequential because the architecture is straightforward: one set of inputs, one output.

---

### Q13. What does `model.compile()` do? Explain each parameter.

**Answer:**

`compile()` configures the model for training. It sets three things: how to optimise, how to measure error, and what to monitor.

```python
model.compile(
    optimizer='adam',   # HOW to update weights
    loss='mse',         # WHAT to minimise
    metrics=['mae']     # WHAT to display during training
)
```

**optimizer — Adam:**
- Stands for Adaptive Moment Estimation
- Automatically adjusts the learning rate for each weight
- Combines the best of two older optimisers (RMSprop + momentum)
- Default choice for most problems — works well without tuning

```python
# With custom learning rate:
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse')
```

**loss — MSE:**
- The function backpropagation uses to calculate gradients
- This is what the model actually minimises

**metrics — MAE:**
- Only for humans to read during training
- Does NOT affect weight updates
- MAE is shown in dollar terms, easier to interpret than MSE

**Output during training:**
```
Epoch 1/30
8/8 ━━━━━━━━━━━━ loss: 341424 - mae: 562 - val_loss: 367111 - val_mae: 586
       ↑                ↑            ↑            ↑                ↑
  progress        training     training      validation       validation
   bar            loss         MAE           loss             MAE
```

---

### Q14. What does `model.fit()` do? Explain epochs and batch size.

**Answer:**

`model.fit()` runs the training loop — forward pass, loss calculation, backpropagation, weight update.

```python
history = model.fit(
    X_train_scaled, y_train,   # training data
    epochs=30,                  # how many full passes through data
    batch_size=16,              # samples per weight update
    validation_split=0.2        # hold out 20% for validation
)
```

**Epoch:**
One complete pass through the **entire** training dataset.
- Epoch 1: Model sees all 128 training samples, updates weights
- Epoch 2: Same data again, updates weights again
- Epoch 30: By now the model has learned the patterns

In our project: 30 epochs × 128 samples = 3,840 training steps total.

**Batch Size:**
How many samples to process before updating weights.

```
Total training samples: 128
Batch size: 16

Batches per epoch = 128 / 16 = 8 batches
Each epoch: 8 small weight updates
```

| Batch Size | Pros | Cons |
|-----------|------|------|
| Small (8-16) | Better generalisation, less memory | Noisy, slow |
| Large (256-512) | Faster training, stable | More memory, may overfit |
| Full dataset | Stable gradients | Very slow, memory intensive |

**`validation_split=0.2`:**
Holds out 20% of training data to monitor overfitting. This data is NOT used for weight updates — only to check performance on unseen data.

---

### Q15. What is the `history` object returned by `model.fit()`?

**Answer:**

`history` is a dictionary of metric values recorded at each epoch. You use it to plot training curves and diagnose problems.

```python
history = model.fit(X_train, y_train, epochs=30, validation_split=0.2)

# Keys available:
print(history.history.keys())
# → ['loss', 'mae', 'val_loss', 'val_mae']

# Access values:
print(history.history['loss'])     # list of 30 training loss values
print(history.history['val_loss']) # list of 30 validation loss values
```

**In our project — plotting the history:**
```python
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Model Loss Over Time')
plt.legend()
```

**What the chart reveals:**
- Both lines decreasing together → healthy training
- Training loss low but val_loss high → overfitting
- Both losses high and flat → underfitting (model not learning)

In our project the loss went from ~341,000 down to ~9,800 — a 97% improvement over 30 epochs.

---

## 5. Project Deep Dive

---

### Q16. Walk me through your house price prediction project end to end.

**Answer:**

This is a supervised regression problem. The goal was to predict house prices given three features: size, bedrooms, and age.

**Step 1 — Data Generation:**
```python
# No real dataset — we generated synthetic data
size     = np.random.uniform(800, 4000, 200)
bedrooms = np.random.randint(1, 6, 200)
age      = np.random.uniform(0, 50, 200)
price    = 100 + (size * 0.15) + (bedrooms * 50) - (age * 1.5) + noise
```
I added Gaussian noise to make the data realistic — real houses don't follow a perfect formula.

**Step 2 — Data Preparation:**
- Split: 160 samples training, 40 samples testing (80/20 split)
- Normalised all features with StandardScaler so the neural network trains stably

**Step 3 — Model Architecture:**
```
Input (3 features) → Dense(64, ReLU) → Dense(32, ReLU) → Dense(16, ReLU) → Dense(1, linear)
```
Total: 2,881 trainable parameters.

**Step 4 — Training:**
- Optimizer: Adam
- Loss: MSE (regression task)
- Epochs: 30, Batch size: 16
- Loss dropped from 341,424 to 9,801

**Step 5 — Evaluation:**
- Test MAE: $71.50k (average prediction error)
- Predictions vs actual scatter plot showed good alignment

**Step 6 — Inference:**
- Loaded saved model and predicted on 6 custom houses
- Performed sensitivity analysis: price per bedroom, per sqft, per year of age

**Key insight from the demo results:**
- +1 bedroom adds ~$41k
- +1000 sqft adds ~$122k
- +10 years of age reduces price by ~$12k

---

### Q17. Why did you use 64 → 32 → 16 neurons? Why not 10 → 10 → 10?

**Answer:**

The funnel-shaped architecture (gradually reducing neurons) is a common best practice, not a strict rule.

**Reasoning:**

The first layer needs enough capacity to capture all the raw feature relationships. With 3 features (size, bedrooms, age) and their interactions (e.g., large + new + many bedrooms = expensive), 64 neurons gives enough "workspace".

As we go deeper, the model builds higher-level representations. The later layers combine what the earlier layers learned, so they need fewer neurons.

**Think of it like compression:**
```
Layer 1 (64 neurons): "I see large size, 3 bedrooms, new house"
Layer 2 (32 neurons): "This combination suggests premium segment"
Layer 3 (16 neurons): "Premium segment → high price range"
Layer 4 (1 neuron):   "$620k"
```

**What would happen with 10→10→10?**
- Might underfit — too few neurons to learn complex patterns
- Less representation capacity

**What about 512→256→128→64?**
- Would likely overfit on our small 200-sample dataset
- Takes much longer to train
- Overkill for 3 input features

**Rule of thumb:** Start small, add complexity only if the model underfits.

---

### Q18. How did you save and reload the trained model?

**Answer:**

Saving a model means preserving its architecture AND the learned weights, so you don't have to retrain from scratch every time.

**Saving:**
```python
# HDF5 format (.h5) — legacy, still widely used
model.save('house_price_model.h5')

# Native Keras format (.keras) — recommended in Keras 3+
model.save('house_price_model.keras')
```

**Loading:**
```python
import tensorflow as tf

# For inference only (no retraining needed)
model = tf.keras.models.load_model('house_price_model.h5', compile=False)

# For continued training (need the optimizer state too)
model = tf.keras.models.load_model('house_price_model.h5')
```

**We used `compile=False` in the demo because:**
- We only needed `model.predict()` — no training
- Avoids a serialization compatibility issue with the 'mse' string in older Keras HDF5 files

**Making predictions after loading:**
```python
custom_house = np.array([[2000, 3, 10]])   # 2000 sqft, 3 beds, 10yr old
scaled       = scaler.transform(custom_house)
price        = model.predict(scaled, verbose=0)
print(f"${price[0][0]:,.0f}k")             # → $428k
```

**Important:** You must also save/recreate the **StandardScaler** — the model expects scaled inputs!

---

### Q19. What is `model.summary()` and what does it tell you?

**Answer:**

`model.summary()` prints a table showing the architecture, output shapes, and parameter counts.

**Our project output:**
```
Model: "sequential"
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)  ┃ Output Shape     ┃ Param #       ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense (Dense) │ (None, 64)       │   256         │
│ dense_1       │ (None, 32)       │ 2,080         │
│ dense_2       │ (None, 16)       │   528         │
│ dense_3       │ (None, 1)        │    17         │
└───────────────┴──────────────────┴───────────────┘
 Total params: 2,881
```

**How parameter counts are calculated:**

Layer 1: 3 inputs × 64 neurons + 64 biases = **256 params**
```
weights: 3 × 64 = 192
biases:  1 × 64 = 64
total:   256
```

Layer 2: 64 inputs × 32 neurons + 32 biases = **2,080 params**
```
weights: 64 × 32 = 2048
biases:  1  × 32 = 32
total:   2,080
```

Layer 3: 32 × 16 + 16 = **528 params**

Layer 4: 16 × 1 + 1 = **17 params**

**Total: 256 + 2,080 + 528 + 17 = 2,881 parameters**

`(None, 64)` means the batch size (None) is flexible — the model accepts any number of samples at once.

---

## 6. Model Evaluation & Improvement

---

### Q20. What is MAE and MSE? Which is better?

**Answer:**

Both measure the average prediction error, but in different ways.

**MAE — Mean Absolute Error:**
```
MAE = (1/n) × Σ |predicted - actual|
```
- Average of absolute differences
- In our project: **$71.50k** — predictions are off by $71.5k on average
- Less sensitive to outliers
- Same unit as the target (dollars)

**MSE — Mean Squared Error:**
```
MSE = (1/n) × Σ (predicted - actual)²
```
- Squares the errors — punishes large mistakes more heavily
- Not in the same unit as the target (dollars²)
- More sensitive to outliers

**RMSE — Root Mean Squared Error:**
```
RMSE = √MSE
```
- Back in the same units as the target
- Between MAE and MSE in sensitivity

**Which to use?**

| Situation | Recommended metric |
|-----------|-------------------|
| Large errors are very costly | MSE or RMSE |
| Outliers in data | MAE |
| Need interpretability | MAE or RMSE |
| Used as loss function | MSE (better gradients) |

We used MSE as the loss function (drives training) and MAE as the metric (human-readable reporting).

---

### Q21. Our model had MAE of $71.5k. Is that good or bad?

**Answer:**

This is a great interview question about **contextual evaluation** — a number means nothing without context.

**Our dataset price range:** $190k – $875k

**MAE as % of price range:** 71.5 / (875 - 190) = ~10.4%

**As % of average price (~$500k):** 71.5 / 500 = ~14.3%

**Context:**

For a beginner project with:
- Only 200 data points (very small)
- Synthetic / noisy data
- Only 3 features (real house prices depend on 100+ factors)
- Only 30 training epochs

A 14% error is **acceptable for learning**, but **not production-ready**.

**How to improve:**
1. More training data (thousands of real house records)
2. More features (location, garage, school district, etc.)
3. More epochs (100-200)
4. Hyperparameter tuning (try different architectures)
5. Feature engineering (e.g., price per sqft as a derived feature)

**In real estate ML products:** Top models achieve 3-5% MAE using hundreds of features and millions of data points.

---

### Q22. How would you improve this model?

**Answer:**

**1. Add more features:**
```python
# Current: [size, bedrooms, age]
# Better:  [size, bedrooms, age, bathrooms, location_score, lot_size, garage, school_rating]
```

**2. Feature engineering:**
```python
# Derived features often help
price_per_sqft  = price / size
bed_bath_ratio  = bedrooms / bathrooms
is_new          = 1 if age < 5 else 0    # binary flag
```

**3. Add more training data:**
- 200 samples is very small
- 10,000+ samples would significantly improve accuracy

**4. Tune hyperparameters:**
```python
# Try different architectures
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)
])
```

**5. Add callbacks to prevent overfitting:**
```python
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,         # stop if no improvement for 5 epochs
    restore_best_weights=True
)

model.fit(X_train, y_train, epochs=200, callbacks=[early_stop])
```

**6. Use cross-validation:**
```python
# Instead of a single 80/20 split, use 5-fold cross-validation
# for a more reliable estimate of performance
from sklearn.model_selection import KFold
```

---

## 7. Advanced Topics

---

### Q23. What is the learning rate and why does it matter?

**Answer:**

The learning rate controls how big each weight update step is during training.

```
new_weight = old_weight - (learning_rate × gradient)
```

**Learning rate too high:**
- Weight updates are too large
- Model overshoots the minimum, loss oscillates or explodes
- Training is unstable

**Learning rate too low:**
- Weight updates are tiny
- Training is very slow
- May get stuck in local minima

**Learning rate just right:**
- Smooth decrease in loss
- Converges efficiently

**Adam optimizer in our project:**
Adam adapts the learning rate automatically per parameter — which is why it's the default choice. Default lr = 0.001.

```python
# Experimenting with learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.01)   # 10x larger
optimizer = keras.optimizers.Adam(learning_rate=0.0001) # 10x smaller
model.compile(optimizer=optimizer, loss='mse')
```

**Learning rate scheduling** — reduce lr over time:
```python
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=100,
    decay_rate=0.9
)
```

---

### Q24. What is a tensor? Give examples from our project.

**Answer:**

A tensor is a multi-dimensional array. All data in TensorFlow flows as tensors.

| Tensor Rank | Name | Shape example | In our project |
|------------|------|--------------|----------------|
| 0 | Scalar | `()` | Single price: `428.0` |
| 1 | Vector | `(3,)` | One house's features: `[2000, 3, 10]` |
| 2 | Matrix | `(160, 3)` | All training features |
| 3 | 3D Tensor | `(batch, height, width)` | Image batch |
| 4 | 4D Tensor | `(batch, h, w, channels)` | Color image batch |

**In our project:**
```python
X_train_scaled.shape   # → (160, 3)  — 2D tensor, rank 2
y_train.shape          # → (160,)    — 1D tensor, rank 1
y_pred.shape           # → (40, 1)   — 2D tensor, output of model.predict()
```

**TensorFlow operations:**
```python
import tensorflow as tf

a = tf.constant([[1, 2, 3], [4, 5, 6]])  # shape (2, 3)
b = tf.constant([[1], [2], [3]])          # shape (3, 1)

tf.matmul(a, b)  # Matrix multiplication → shape (2, 1)
```

Inside the neural network, every layer performs tensor operations (matrix multiplications) on your data.

---

### Q25. Explain Dropout. Why does randomly turning off neurons help?

**Answer:**

Dropout is a regularisation technique that randomly sets a fraction of neurons to zero during each training step.

```python
keras.layers.Dropout(0.2)   # 20% of neurons randomly set to 0
```

**Why it prevents overfitting:**

Without dropout, neurons can develop **co-adaptations** — neuron A always relies on neuron B to correct its mistakes. They work together to memorise training data but fail on new data.

With dropout, each neuron must learn to be useful independently, because its "partner" might be switched off at any moment. This forces the network to learn more robust, general patterns.

**Analogy:** In a sports team, if one player is randomly sitting out each practice, every other player has to be capable of multiple roles. The team becomes more resilient.

**Important:** Dropout is only active during **training**. During `model.predict()`, all neurons are active, but their outputs are scaled by the dropout rate.

```python
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.2),    # ← training only
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])
```

---

### Q26. What is the difference between `model.predict()` and `model.evaluate()`?

**Answer:**

| | `model.predict()` | `model.evaluate()` |
|--|------------------|-------------------|
| **Returns** | The actual predictions (numbers) | Loss + metrics (performance scores) |
| **Used for** | Getting predictions on new data | Checking how good the model is |
| **In our project** | `y_pred = model.predict(X_test_scaled)` | `loss, mae = model.evaluate(X_test_scaled, y_test)` |

```python
# predict() — returns the house price predictions
y_pred = model.predict(X_test_scaled)
print(y_pred[:3])
# → [[450.2], [328.7], [612.4]]   ← actual price predictions

# evaluate() — returns the loss and metrics
test_loss, test_mae = model.evaluate(X_test_scaled, y_test)
print(f"MSE: {test_loss:.2f}, MAE: {test_mae:.2f}")
# → MSE: 7210.70, MAE: 71.50
```

**When to use each:**
- Use `evaluate()` to get a single performance number (for reports, comparisons)
- Use `predict()` when you need the actual prediction values (for deployment, analysis)

---

## 8. Scenario / Problem-Solving Questions

---

### Q27. Your model's training loss is 500 but validation loss is 50,000. What do you do?

**Answer:**

This is a classic **overfitting** scenario. The model memorised training data but doesn't generalise.

**Diagnosis:**
```
Training loss:   500    (model fits training data very well)
Validation loss: 50,000 (model fails on unseen data)
Gap:             49,500 (extremely large — severe overfitting)
```

**Solutions (in order of what to try first):**

1. **Add Dropout:**
```python
keras.layers.Dropout(0.3)   # 30% neurons randomly dropped
```

2. **Add L2 Regularisation:**
```python
keras.layers.Dense(64, activation='relu', kernel_regularizer='l2')
```

3. **Get more training data** — most effective solution

4. **Simplify the model** — fewer layers/neurons

5. **Early stopping:**
```python
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
model.fit(..., callbacks=[early_stop])
```

6. **Data augmentation** — artificially increase dataset size by adding slightly modified copies

---

### Q28. Someone gives you a real housing dataset with 50 features. How would you approach it?

**Answer:**

**Step 1 — Exploratory Data Analysis (EDA):**
```python
import pandas as pd
df = pd.read_csv('houses.csv')
df.info()         # check data types, null values
df.describe()     # statistics (mean, std, min, max)
df.isnull().sum() # count missing values per column
```

**Step 2 — Handle missing values:**
```python
# Fill numerical missing values with median
df['size'].fillna(df['size'].median(), inplace=True)

# Drop rows where target is missing
df.dropna(subset=['price'], inplace=True)
```

**Step 3 — Encode categorical variables:**
```python
# 'neighbourhood' is text — convert to numbers
df = pd.get_dummies(df, columns=['neighbourhood'])
```

**Step 4 — Feature selection:**
```python
# Check correlation with target
correlation = df.corr()['price'].sort_values(ascending=False)
print(correlation.head(10))  # top 10 features correlated with price
```

**Step 5 — Remove highly correlated features (multicollinearity):**
If `total_rooms` and `bedrooms` are 95% correlated, keeping both adds noise.

**Step 6 — Scale and train as before**

**Step 7 — Use cross-validation** for more reliable evaluation

This kind of methodical data preparation often matters more than the model architecture itself.

---

### Q29. How would you deploy this model so others can use it?

**Answer:**

**Option 1 — FastAPI web service (most common):**
```python
from fastapi import FastAPI
import tensorflow as tf, numpy as np
from sklearn.preprocessing import StandardScaler

app   = FastAPI()
model = tf.keras.models.load_model('house_price_model.h5', compile=False)

@app.post("/predict")
def predict(size: float, bedrooms: int, age: float):
    features = scaler.transform([[size, bedrooms, age]])
    price    = model.predict(features, verbose=0)[0][0]
    return {"predicted_price_k": round(float(price), 2)}
```

A user calls: `POST /predict?size=2000&bedrooms=3&age=10` → `{"predicted_price_k": 428.53}`

**Option 2 — TensorFlow Serving (production-grade):**
```bash
# Export model in SavedModel format
model.save('saved_model/house_price')

# Serve with TF Serving Docker container
docker run -p 8501:8501 \
  -v $(pwd)/saved_model:/models/house_price \
  -e MODEL_NAME=house_price \
  tensorflow/serving
```

**Option 3 — TensorFlow Lite (mobile/edge):**
```python
converter  = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open('model.tflite', 'wb').write(tflite_model)
```

**Option 4 — Streamlit (quick demo app):**
```python
import streamlit as st

size     = st.slider('House Size (sqft)', 800, 4000, 2000)
bedrooms = st.selectbox('Bedrooms', [1,2,3,4,5])
age      = st.slider('Age (years)', 0, 50, 10)

if st.button('Predict Price'):
    features = scaler.transform([[size, bedrooms, age]])
    price    = model.predict(features, verbose=0)[0][0]
    st.success(f'Predicted Price: ${price:,.0f}k')
```

---

### Q30. What would you change if you had to predict whether a house is "affordable" or "luxury"?

**Answer:**

This changes the problem from **regression** to **binary classification**. Three things must change:

**1 — Target variable:**
```python
# Regression target: price = $428,000 (continuous)
# Classification target: label = 0 (affordable) or 1 (luxury)

threshold = 500   # $500k cut-off
y_class   = (y > threshold).astype(int)   # 0 or 1
```

**2 — Output layer and loss function:**
```python
# Regression (current):
keras.layers.Dense(1)                           # linear output
model.compile(loss='mse', metrics=['mae'])

# Binary Classification (new):
keras.layers.Dense(1, activation='sigmoid')     # output: 0.0–1.0 probability
model.compile(loss='binary_crossentropy', metrics=['accuracy'])
```

**3 — Interpreting predictions:**
```python
# Regression:
price = model.predict([[2000, 3, 10]])    # → [[428.5]]

# Classification:
prob  = model.predict([[2000, 3, 10]])    # → [[0.62]]  (62% chance of luxury)
label = (prob > 0.5).astype(int)         # → 0 (affordable) or 1 (luxury)
```

**Evaluation metrics change too:**
```python
# Instead of MAE, you'd use:
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

print(accuracy_score(y_test, y_pred_labels))
print(confusion_matrix(y_test, y_pred_labels))
print(classification_report(y_test, y_pred_labels))
```

This question tests whether you understand that the model architecture is driven by the **problem type**, not preference.

---

## Quick Reference Cheat Sheet

```
DATA PIPELINE
─────────────────────────────────────────────
Load data          → pandas / numpy
Split              → train_test_split (80/20)
Scale features     → StandardScaler (fit on train only!)
Encode categories  → pd.get_dummies or LabelEncoder

MODEL BUILDING
─────────────────────────────────────────────
Architecture       → keras.Sequential
Layers             → keras.layers.Dense
Activation (hidden)→ relu
Activation (output)→ linear (regression), sigmoid (binary), softmax (multi)

COMPILE
─────────────────────────────────────────────
Optimizer          → adam (default)
Loss (regression)  → mse
Loss (binary clf)  → binary_crossentropy
Loss (multi clf)   → categorical_crossentropy
Metric             → mae (regression), accuracy (classification)

TRAINING
─────────────────────────────────────────────
model.fit()        → trains the model
epochs             → full passes through dataset
batch_size         → samples per weight update
validation_split   → fraction held out to monitor overfitting

EVALUATION
─────────────────────────────────────────────
model.evaluate()   → returns loss + metrics
model.predict()    → returns actual predictions

SAVE / LOAD
─────────────────────────────────────────────
model.save('model.keras')
model = tf.keras.models.load_model('model.keras')
```

---

*Good luck with your interview! The best preparation is to re-run this project, change one thing at a time, and observe what happens.*
