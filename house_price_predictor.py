"""
Simple TensorFlow Neural Network for House Price Prediction
This project demonstrates the basics of machine learning with TensorFlow.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Generate synthetic housing dataset
def generate_house_data(num_samples=200):
    """
    Create synthetic house data with features:
    - size (sq ft): 800-4000
    - bedrooms: 1-5
    - age (years): 0-50
    - target: price in thousands
    """
    size = np.random.uniform(800, 4000, num_samples)
    bedrooms = np.random.randint(1, 6, num_samples)
    age = np.random.uniform(0, 50, num_samples)

    # Synthetic price calculation (realistic formula)
    # price = base + (size * rate) + (bedrooms * increment) - (age * depreciation)
    price = 100 + (size * 0.15) + (bedrooms * 50) - (age * 1.5) + np.random.normal(0, 30, num_samples)
    price = np.maximum(price, 50)  # Ensure minimum price of 50k

    features = np.column_stack([size, bedrooms, age])
    return features, price

print("=" * 60)
print("TENSORFLOW HOUSE PRICE PREDICTION PROJECT")
print("=" * 60)

# Step 1: Generate and prepare data
print("\n[Step 1] Generating synthetic housing data...")
X, y = generate_house_data(num_samples=200)
print(f"Generated {X.shape[0]} house samples with {X.shape[1]} features each")
print(f"Features: [Size (sq ft), Bedrooms, Age (years)]")
print(f"Price range: ${y.min():.2f}k - ${y.max():.2f}k")

# Split data into training and testing sets
print("\n[Step 2] Splitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normalize features (important for neural networks!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")

# Step 2: Build the neural network model
print("\n[Step 3] Building neural network model...")
model = keras.Sequential([
    keras.layers.Input(shape=(3,)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)  # Single output: price
])

print("Model Architecture:")
model.summary()

# Step 3: Compile the model
print("\n[Step 4] Compiling model...")
model.compile(
    optimizer='adam',  # Adaptive learning rate optimization
    loss='mse',  # Mean Squared Error for regression
    metrics=['mae']  # Mean Absolute Error for evaluation
)

# Step 4: Train the model
print("\n[Step 5] Training model (30 epochs)...")
history = model.fit(
    X_train_scaled, y_train,
    epochs=30,
    batch_size=16,
    validation_split=0.2,
    verbose=1
)

# Step 5: Evaluate on test set
print("\n[Step 6] Evaluating on test set...")
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"Test Loss (MSE): {test_loss:.4f}")
print(f"Test MAE: ${test_mae:.2f}k")

# Step 6: Make predictions
print("\n[Step 7] Making predictions...")
y_pred = model.predict(X_test_scaled, verbose=0)

# Show some example predictions
print("\nSample Predictions vs Actual:")
print("-" * 70)
print(f"{'Size(sqft)':<12} {'Beds':<8} {'Age(yrs)':<10} {'Predicted':<15} {'Actual':<15}")
print("-" * 70)
for i in range(min(5, len(X_test))):
    size, beds, age = X_test[i]
    pred_price = y_pred[i][0]
    actual_price = y_test[i]
    error = abs(pred_price - actual_price)
    print(f"{size:<12.0f} {beds:<8.0f} {age:<10.1f} ${pred_price:<14.2f}k ${actual_price:<14.2f}k")

# Calculate accuracy metrics
mse = np.mean((y_pred.flatten() - y_test) ** 2)
mae = np.mean(np.abs(y_pred.flatten() - y_test))
print("-" * 70)
print(f"Mean Absolute Error (MAE): ${mae:.2f}k")
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Step 7: Test with custom input
print("\n[Step 8] Testing with custom house...")
custom_house = np.array([[2000, 3, 10]])  # 2000 sqft, 3 beds, 10 years old
custom_house_scaled = scaler.transform(custom_house)
custom_prediction = model.predict(custom_house_scaled, verbose=0)
print(f"House specs: 2000 sq ft, 3 bedrooms, 10 years old")
print(f"Predicted price: ${custom_prediction[0][0]:.2f}k (${custom_prediction[0][0]*1000:.0f})")

# Plot training history
print("\n[Step 9] Generating training visualization...")
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Model Loss Over Time')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual Price ($k)')
plt.ylabel('Predicted Price ($k)')
plt.title('Predictions vs Actual')
plt.grid(True)

plt.tight_layout()
plt.savefig('/Users/umamalav/Documents/Projects/ML-AI/training_results.png', dpi=100)
print("✓ Visualization saved to training_results.png")

# Save the model
print("\n[Step 10] Saving trained model...")
model.save('/Users/umamalav/Documents/Projects/ML-AI/house_price_model.h5')
print("✓ Model saved to house_price_model.h5")

print("\n" + "=" * 60)
print("PROJECT COMPLETE!")
print("=" * 60)
