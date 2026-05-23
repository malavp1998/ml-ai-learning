# TensorFlow Learning Project: House Price Predictor

A beginner-friendly project to learn TensorFlow by building a neural network that predicts house prices.

> **Interview Prep?** See [INTERVIEW_QA.md](INTERVIEW_QA.md) — 30 questions with detailed answers built around this project, from absolute basics to advanced topics.

## 📚 What is TensorFlow?

**TensorFlow** is an open-source machine learning framework developed by Google. It enables you to build and train deep learning models (neural networks) for various tasks like:
- **Classification** (predicting categories)
- **Regression** (predicting continuous values)
- **Computer Vision** (image recognition)
- **Natural Language Processing** (text analysis)

### Key Concepts

#### 1. **Neural Networks**
A neural network is inspired by how the brain works. It consists of:
- **Neurons** (nodes): Small units that process information
- **Layers**: Groups of neurons stacked together
- **Connections**: Weighted paths between neurons that learn patterns

```
Input Layer → Hidden Layers → Output Layer
  (Features)   (Learn patterns)  (Prediction)
```

#### 2. **Tensors**
The fundamental data structure in TensorFlow:
- **Scalar**: Single number (e.g., `5`)
- **Vector**: 1D array (e.g., `[1, 2, 3]`)
- **Matrix**: 2D array (e.g., `[[1, 2], [3, 4]]`)
- **Tensor**: Multi-dimensional array (e.g., images with height, width, color channels)

#### 3. **Training Process**
```
1. Initialize: Create a model with random weights
2. Forward Pass: Feed data through the network
3. Calculate Loss: Measure how wrong predictions are
4. Backpropagation: Adjust weights to reduce loss
5. Repeat: Do this many times until model improves
```

#### 4. **Key Components**

| Component | Purpose | Example |
|-----------|---------|---------|
| **Input Layer** | Accept raw features | House size, bedrooms, age |
| **Hidden Layers** | Learn complex patterns | Dense layers with 64 → 32 → 16 neurons |
| **Output Layer** | Make predictions | Single value (house price) |
| **Activation Function** | Add non-linearity | ReLU, Sigmoid, Tanh |
| **Loss Function** | Measure prediction error | Mean Squared Error (MSE) |
| **Optimizer** | Update weights to reduce loss | Adam, SGD |

---

## 🏠 Project: House Price Prediction

### What This Project Does

This project trains a neural network to predict house prices based on three features:
- **Size** (square footage): 800-4000 sq ft
- **Bedrooms**: 1-5 bedrooms
- **Age**: 0-50 years old

The model learns the relationship between these features and house prices.

### Project Structure

```
ML-AI/
├── house_price_predictor.py    # Main training script
├── house_price_model.h5        # Trained model (generated after running)
├── training_results.png        # Visualization (generated after running)
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

### How the Code Works

#### Step 1: Generate Data
```python
X, y = generate_house_data(num_samples=200)
# Creates 200 synthetic houses with random features and prices
```

#### Step 2: Prepare Data
```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
# Normalize features to range [-1, 1] for better training
```

#### Step 3: Build Model
```python
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)  # Output: predicted price
])
```

**What this means:**
- Layer 1: Takes 3 inputs, outputs 64 neurons with ReLU activation
- Layer 2: Takes 64 inputs, outputs 32 neurons
- Layer 3: Takes 32 inputs, outputs 16 neurons
- Layer 4: Takes 16 inputs, outputs 1 value (the price)

#### Step 4: Compile Model
```python
model.compile(
    optimizer='adam',        # Learning strategy
    loss='mse',             # How to measure error
    metrics=['mae']         # How to evaluate performance
)
```

#### Step 5: Train Model
```python
model.fit(X_train_scaled, y_train, epochs=30, batch_size=16)
# Train for 30 complete passes through the data
# Process 16 samples at a time for efficiency
```

#### Step 6: Make Predictions
```python
y_pred = model.predict(X_test_scaled)
# Use trained model to predict prices for new houses
```

---

## 🚀 Installation & Running

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Setup

1. **Create requirements.txt**
   ```
   tensorflow>=2.10.0
   numpy>=1.20.0
   scikit-learn>=1.0.0
   matplotlib>=3.3.0
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the project**
   ```bash
   python house_price_predictor.py
   ```

### Expected Output

```
============================================================
TENSORFLOW HOUSE PRICE PREDICTION PROJECT
============================================================

[Step 1] Generating synthetic housing data...
Generated 200 house samples with 3 features each

[Step 2] Splitting data (80% train, 20% test)...
Training samples: 160, Test samples: 40

[Step 3] Building neural network model...
Model Architecture:
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 input_layer (Dense)         (None, 64)                256
 hidden_layer_1 (Dense)      (None, 32)                2080
 hidden_layer_2 (Dense)      (None, 16)                528
 output_layer (Dense)        (None, 1)                 17
=================================================================
Total params: 2,881

[Step 5] Training model (30 epochs)...
Epoch 1/30
5/5 [==============================] 0s 2ms/step - loss: 5347.8477 - mae: 70.3481 - val_loss: 4891.8340 - val_mae: 65.0980
...
Epoch 30/30
5/5 [==============================] 0s 2ms/step - loss: 215.4859 - mae: 11.2546 - val_loss: 289.5934 - val_mae: 13.4521

[Step 6] Evaluating on test set...
Test Loss (MSE): 287.34
Test MAE: $13.27k

[Step 7] Making predictions...
Sample Predictions vs Actual:
----------------------------------------------------------------------
Size(sqft)   Beds     Age(yrs)   Predicted       Actual
----------------------------------------------------------------------
 2145         3        8.5        $445.32k        $438.50k
 1876         2        22.4       $395.81k        $410.21k

[Step 8] Testing with custom house...
House specs: 2000 sq ft, 3 bedrooms, 10 years old
Predicted price: $446.32k ($446320)

[Step 9] Generating training visualization...
✓ Visualization saved to training_results.png

[Step 10] Saving trained model...
✓ Model saved to house_price_model.h5

============================================================
PROJECT COMPLETE!
============================================================
```

### Output Files Generated

1. **training_results.png**: Visualization showing:
   - Training loss curve (how the model improved)
   - Predictions vs actual values scatter plot

2. **house_price_model.h5**: Trained model file (can be loaded and used later)

---

## 📊 Understanding the Results

### Metrics Explained

- **Loss (MSE)**: Measures how far predictions are from actual values
  - Lower is better
  - MSE = mean of squared differences
  
- **MAE (Mean Absolute Error)**: Average error in price prediction
  - If MAE = $13.27k, predictions are off by about $13,270 on average

### Training Curve
- The loss should decrease over epochs
- If loss increases, the model is overfitting
- If loss plateaus, you can stop training earlier

---

## 🔧 Experimenting & Learning

### Try These Modifications

#### 1. **Change Architecture** (More/Fewer Layers)
```python
# Simpler model (fewer neurons)
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(3,)),
    keras.layers.Dense(1)
])

# More complex model (deeper)
model = keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=(3,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
])
```

#### 2. **Add Dropout** (Prevent Overfitting)
```python
model = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(3,)),
    keras.layers.Dropout(0.2),  # Randomly turn off 20% of neurons
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(1)
])
```

#### 3. **Change Optimizer**
```python
model.compile(
    optimizer='sgd',  # Stochastic Gradient Descent
    loss='mse',
    metrics=['mae']
)
```

#### 4. **Change Learning Rate**
```python
optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
```

#### 5. **Increase Epochs**
```python
history = model.fit(
    X_train_scaled, y_train,
    epochs=100,  # Train longer
    batch_size=16,
    validation_split=0.2
)
```

---

## 🎯 Key Takeaways

1. **Neural networks learn by adjusting weights** through backpropagation
2. **Normalization is crucial** for stable training
3. **Architecture matters** - too simple and model underfits, too complex and it overfits
4. **Monitor training loss** to ensure the model is learning
5. **Separate train/test data** to truly evaluate performance
6. **Activation functions add non-linearity** allowing networks to learn complex patterns

---

## 📚 Next Steps

### Intermediate Projects
- **MNIST Digit Classification**: Classify handwritten digits (image classification)
- **Stock Price Prediction**: Use LSTM networks for time series
- **Sentiment Analysis**: Classify text as positive/negative

### Advanced Topics
- **Convolutional Neural Networks (CNN)**: For image processing
- **Recurrent Neural Networks (RNN/LSTM)**: For sequences
- **Transfer Learning**: Use pre-trained models
- **Hyperparameter Tuning**: Automated optimization

### Useful Resources
- [TensorFlow Official Tutorial](https://tensorflow.org/tutorials)
- [Keras API Reference](https://keras.io/api/)
- [Andrew Ng's Machine Learning Course](https://www.coursera.org/learn/machine-learning)

---

## 📝 Code Walkthrough

### Training Loop Simplified
```
FOR each epoch (1 to 30):
    FOR each batch of 16 samples:
        1. Forward pass: prediction = model(input)
        2. Calculate loss: loss = MSE(prediction, actual)
        3. Backpropagation: calculate gradients
        4. Update weights: weights -= learning_rate * gradients
    
    Print: epoch, loss, validation loss
```

### Why This Works

The neural network learns by finding patterns in the data:
- It discovers that larger homes = higher prices
- More bedrooms = higher prices
- Older homes = lower prices

The model automatically learns these relationships from examples!

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| Import errors | Run `pip install -r requirements.txt` |
| Out of memory | Reduce `batch_size` or `num_samples` |
| Loss not decreasing | Reduce learning rate, change optimizer |
| Very slow training | Reduce epochs or use GPU (if available) |
| Model file not saving | Check write permissions in directory |

---

## 💡 How TensorFlow Works Under the Hood

1. **Graph Execution**: Operations are compiled into a computational graph
2. **Automatic Differentiation**: Gradients calculated automatically
3. **Optimization**: Uses various algorithms (Adam, SGD) to minimize loss
4. **Vectorization**: Uses NumPy/linear algebra for efficiency
5. **GPU Support**: Can run on GPU for 10-100x speedup

---

Happy Learning! 🚀
