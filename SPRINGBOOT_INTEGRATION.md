# Using the Trained Model in Spring Boot

## First — What is ACTUALLY inside house_price_model.h5?

Before integration, you must understand what the model really is. It is **not magic**. It is just **2,881 numbers** (weights and biases) and a set of matrix-multiply + ReLU instructions.

---

### The Prediction Formula (Plain Math)

When you call `model.predict([2000, 3, 10])` here is what literally happens:

```
Input: [size=2000, bedrooms=3, age=10]

Step 1 — Normalize (StandardScaler):
  size_scaled   = (2000 - mean_size)   / std_size   = -0.3706
  beds_scaled   = (3    - mean_beds)   / std_beds   = -0.0244
  age_scaled    = (10   - mean_age)    / std_age    = -1.0687

Step 2 — Layer 1 (3 inputs → 64 neurons):
  For each of 64 neurons:
    z = (size_scaled × w_size) + (beds_scaled × w_beds) + (age_scaled × w_age) + bias
    output = max(0, z)   ← ReLU activation

Step 3 — Layer 2 (64 inputs → 32 neurons):
  Same: z = inputs @ W2 + b2, then ReLU

Step 4 — Layer 3 (32 inputs → 16 neurons):
  Same: z = inputs @ W3 + b3, then ReLU

Step 5 — Layer 4 (16 inputs → 1 output):
  z = inputs @ W4 + b4   ← NO activation (linear output)
  output = 440.83   ← predicted price in $k

Final prediction: $440,838
```

### What is stored in the .h5 file

```
house_price_model.h5
├── model_config (JSON)       ← architecture: 4 Dense layers, ReLU, shapes
├── layer_1/
│   ├── weights  (3×64  = 192 float32 numbers)
│   └── biases   (64         float32 numbers)
├── layer_2/
│   ├── weights  (64×32 = 2048 float32 numbers)
│   └── biases   (32         float32 numbers)
├── layer_3/
│   ├── weights  (32×16 = 512  float32 numbers)
│   └── biases   (16          float32 numbers)
└── layer_4/
    ├── weights  (16×1  = 16   float32 numbers)
    └── biases   (1           float32 number)

Total: 2,881 float32 numbers — that is the entire model
```

---

## Three Ways to Use This Model in Spring Boot

| Approach | Effort | Best for |
|----------|--------|---------|
| **1. Python REST API** | Low | Any Java app, quickest path |
| **2. ONNX + Java** | Medium | No Python server in production |
| **3. TensorFlow Java** | Medium | Full TF Java ecosystem |

---

## Approach 1 — Python REST API (Recommended for Beginners)

Spring Boot calls a Python microservice that hosts the model.

```
Browser/Client
      ↓
Spring Boot App (Java, port 8080)
      ↓  HTTP POST /predict
Python FastAPI (Python, port 8000)
      ↓
house_price_model.h5
      ↓
Predicted price returned
```

### Step 1 — Create the Python prediction server

**File: `model_server.py`**
```python
from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app   = FastAPI()
model = tf.keras.models.load_model('house_price_model.h5', compile=False)

# Re-create scaler with same seed used during training
np.random.seed(42)
size     = np.random.uniform(800, 4000, 200)
bedrooms = np.random.randint(1, 6, 200)
age      = np.random.uniform(0, 50, 200)
scaler   = StandardScaler().fit(np.column_stack([size, bedrooms, age]))

class HouseFeatures(BaseModel):
    size: float       # square footage
    bedrooms: int     # number of bedrooms
    age: float        # age in years

@app.post("/predict")
def predict_price(house: HouseFeatures):
    features = np.array([[house.size, house.bedrooms, house.age]])
    scaled   = scaler.transform(features)
    price_k  = float(model.predict(scaled, verbose=0)[0][0])
    return {
        "predicted_price_k":   round(price_k, 2),
        "predicted_price_usd": round(price_k * 1000, 0),
        "inputs": {
            "size": house.size,
            "bedrooms": house.bedrooms,
            "age": house.age
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "model": "house_price_v1"}
```

**Install and run:**
```bash
pip install fastapi uvicorn
uvicorn model_server:app --port 8000
```

**Test it works:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"size": 2000, "bedrooms": 3, "age": 10}'

# Response:
# {"predicted_price_k": 440.84, "predicted_price_usd": 440840.0}
```

---

### Step 2 — Spring Boot Project

**pom.xml — only one dependency needed:**
```xml
<dependencies>
    <dependency>
        <groupId>org.springframework.boot</groupId>
        <artifactId>spring-boot-starter-web</artifactId>
    </dependency>
</dependencies>
```

---

### Step 3 — Spring Boot Code

**`HouseRequest.java`** — incoming request from user
```java
package com.example.houseprice.model;

public class HouseRequest {
    private double size;      // square footage
    private int    bedrooms;
    private double age;

    // Getters and Setters
    public double getSize()         { return size; }
    public void setSize(double s)   { this.size = s; }
    public int    getBedrooms()     { return bedrooms; }
    public void setBedrooms(int b)  { this.bedrooms = b; }
    public double getAge()          { return age; }
    public void setAge(double a)    { this.age = a; }
}
```

---

**`PredictionResponse.java`** — response back to user
```java
package com.example.houseprice.model;

public class PredictionResponse {
    private double predictedPriceK;    // in thousands
    private double predictedPriceUsd;  // in dollars
    private String message;

    public PredictionResponse(double priceK, double priceUsd) {
        this.predictedPriceK   = priceK;
        this.predictedPriceUsd = priceUsd;
        this.message = String.format("Predicted price: $%,.0f", priceUsd);
    }

    // Getters
    public double getPredictedPriceK()   { return predictedPriceK; }
    public double getPredictedPriceUsd() { return predictedPriceUsd; }
    public String getMessage()           { return message; }
}
```

---

**`MLService.java`** — calls the Python model server
```java
package com.example.houseprice.service;

import com.example.houseprice.model.HouseRequest;
import com.example.houseprice.model.PredictionResponse;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import org.springframework.http.*;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

@Service
public class MLService {

    // URL of the Python prediction server
    private static final String PYTHON_API_URL = "http://localhost:8000/predict";

    private final RestTemplate  restTemplate = new RestTemplate();
    private final ObjectMapper  objectMapper = new ObjectMapper();

    public PredictionResponse predict(HouseRequest request) throws Exception {

        // 1. Build request body to send to Python
        String requestBody = String.format(
            "{\"size\": %f, \"bedrooms\": %d, \"age\": %f}",
            request.getSize(), request.getBedrooms(), request.getAge()
        );

        // 2. Set headers
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        // 3. Send POST request to Python server
        HttpEntity<String> entity   = new HttpEntity<>(requestBody, headers);
        ResponseEntity<String> resp = restTemplate.postForEntity(
            PYTHON_API_URL, entity, String.class
        );

        // 4. Parse the JSON response
        JsonNode json       = objectMapper.readTree(resp.getBody());
        double priceK       = json.get("predicted_price_k").asDouble();
        double priceUsd     = json.get("predicted_price_usd").asDouble();

        return new PredictionResponse(priceK, priceUsd);
    }
}
```

---

**`HousePriceController.java`** — REST endpoint for clients
```java
package com.example.houseprice.controller;

import com.example.houseprice.model.HouseRequest;
import com.example.houseprice.model.PredictionResponse;
import com.example.houseprice.service.MLService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
public class HousePriceController {

    @Autowired
    private MLService mlService;

    // POST /api/predict
    // Body: {"size": 2000, "bedrooms": 3, "age": 10}
    @PostMapping("/predict")
    public ResponseEntity<PredictionResponse> predictPrice(
            @RequestBody HouseRequest request) {
        try {
            PredictionResponse response = mlService.predict(request);
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            return ResponseEntity.internalServerError().build();
        }
    }

    // GET /api/health
    @GetMapping("/health")
    public ResponseEntity<String> health() {
        return ResponseEntity.ok("{\"status\": \"Spring Boot is running\"}");
    }
}
```

---

### Full Request → Response Flow

```
User calls Spring Boot:
  POST http://localhost:8080/api/predict
  Body: {"size": 2000, "bedrooms": 3, "age": 10}
          ↓
  HousePriceController.predictPrice()
          ↓
  MLService.predict()
          ↓
  HTTP POST to Python: http://localhost:8000/predict
          ↓
  Python normalises: [-0.3706, -0.0244, -1.0687]
          ↓
  Layer 1: 3 → 64 neurons (ReLU)
          ↓
  Layer 2: 64 → 32 neurons (ReLU)
          ↓
  Layer 3: 32 → 16 neurons (ReLU)
          ↓
  Layer 4: 16 → 1 output  (linear) = 440.84
          ↓
  Python returns: {"predicted_price_k": 440.84, "predicted_price_usd": 440840.0}
          ↓
  Spring Boot returns to user:
  {
    "predictedPriceK":   440.84,
    "predictedPriceUsd": 440840.0,
    "message": "Predicted price: $440,840"
  }
```

---

## Approach 2 — ONNX (No Python Server Needed)

ONNX (Open Neural Network Exchange) is a format that lets you run ML models in any language.

**Step 1 — Convert model to ONNX (run once in Python):**
```python
import tf2onnx, tensorflow as tf, numpy as np

model = tf.keras.models.load_model('house_price_model.h5', compile=False)

# Define the input signature
input_sig = [tf.TensorSpec([None, 3], tf.float32, name='input')]

# Convert to ONNX
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=input_sig)

with open('house_price.onnx', 'wb') as f:
    f.write(model_proto.SerializeToString())

print("Saved house_price.onnx — copy this to your Spring Boot resources/")
```

```bash
pip install tf2onnx
python convert_to_onnx.py
```

**Step 2 — Add ONNX Runtime to pom.xml:**
```xml
<dependency>
    <groupId>com.microsoft.onnxruntime</groupId>
    <artifactId>onnxruntime</artifactId>
    <version>1.17.0</version>
</dependency>
```

**Step 3 — Java code loads and runs .onnx directly:**
```java
import ai.onnxruntime.*;
import java.nio.FloatBuffer;

@Service
public class OnnxMLService {

    private final OrtEnvironment env;
    private final OrtSession     session;

    // Scaler stats from training (hardcode these from your Python scaler)
    private final float[] MEAN = {2411.8f, 2.98f, 24.9f};
    private final float[] STD  = {917.4f,  1.41f, 14.5f};

    public OnnxMLService() throws OrtException {
        env     = OrtEnvironment.getEnvironment();
        session = env.createSession("src/main/resources/house_price.onnx");
    }

    public float predict(float size, int bedrooms, float age) throws OrtException {
        // Normalise using the same scaler stats from Python training
        float sizeScaled = (size     - MEAN[0]) / STD[0];
        float bedsScaled = (bedrooms - MEAN[1]) / STD[1];
        float ageScaled  = (age      - MEAN[2]) / STD[2];

        // Create tensor: shape [1, 3] — 1 sample, 3 features
        float[][] input = {{sizeScaled, bedsScaled, ageScaled}};
        OnnxTensor tensor = OnnxTensor.createTensor(env, input);

        // Run inference
        OrtSession.Result result = session.run(
            Collections.singletonMap("input", tensor)
        );

        // Extract predicted price
        float[][] output = (float[][]) result.get(0).getValue();
        return output[0][0];  // price in $k
    }
}
```

**No Python server needed — ONNX runs entirely inside the JVM.**

---

## Approach 3 — TensorFlow Java

```xml
<dependency>
    <groupId>org.tensorflow</groupId>
    <artifactId>tensorflow-core-platform</artifactId>
    <version>0.5.0</version>
</dependency>
```

```bash
# First convert to TF SavedModel format (from Python):
model.save('saved_model_dir')
```

```java
import org.tensorflow.*;
import org.tensorflow.ndarray.*;
import org.tensorflow.types.*;

@Service
public class TFJavaService {

    private final SavedModelBundle bundle;

    public TFJavaService() {
        // Load the SavedModel directory
        bundle = SavedModelBundle.load("saved_model_dir", "serve");
    }

    public float predict(float size, int bedrooms, float age) {
        // Build input tensor [1, 3]
        FloatNdArray inputArray = NdArrays.ofFloats(Shape.of(1, 3));
        inputArray.setFloat(size,            0, 0);
        inputArray.setFloat((float)bedrooms, 0, 1);
        inputArray.setFloat(age,             0, 2);

        TFloat32 inputTensor = TFloat32.tensorOf(inputArray);

        // Run inference
        Map<String, Tensor> inputs  = Map.of("serving_default_input_1:0", inputTensor);
        Map<String, Tensor> outputs = bundle.function("serving_default").call(inputs);

        TFloat32 output = (TFloat32) outputs.get("StatefulPartitionedCall:0");
        return output.getFloat(0, 0);  // price in $k
    }
}
```

---

## Approach Comparison

| | Python REST API | ONNX | TF Java |
|---|---|---|---|
| **Setup effort** | Low | Medium | Medium-High |
| **Performance** | Network latency | Fast (in-process) | Fast (in-process) |
| **Python required** | Yes (as a service) | No | No |
| **Best for** | Prototypes, small scale | Production, microservices | Google-ecosystem production |
| **Scaler handling** | Python handles it | Must port to Java | Must port to Java |

**Recommendation for learning:** Start with Approach 1 (REST API). Once your team understands the flow, migrate to ONNX for production.

---

## Key Concept: The Scaler Problem

This is the most common mistake when integrating ML models into Java:

```
Python training:
  scaler.fit_transform(X)   ← learns mean and std of your dataset

Java inference:
  you MUST apply the exact same normalization before calling the model

If you skip scaling:
  Input [2000, 3, 10] goes into the model as-is
  Model expects numbers in range [-2, +2]
  Numbers like 2000 cause completely wrong predictions
```

**Solution:** Export scaler stats from Python and hardcode in Java:
```python
# Add this to your Python training script after fitting the scaler
print("MEAN:", scaler.mean_)
print("STD: ", scaler.scale_)

# MEAN: [2411.8  2.98  24.9]
# STD:  [917.4   1.41  14.5]
```

```java
// Hardcode in Java
private static final float[] MEAN = {2411.8f, 2.98f, 24.9f};
private static final float[] STD  = {917.4f,  1.41f, 14.5f};

float scaled_size = (size - MEAN[0]) / STD[0];
// ... then pass to model
```

---

## Summary

```
house_price_model.h5 = 2,881 float numbers + architecture description
                        ↓
When you call predict():
  1. Normalize your input (StandardScaler)
  2. Matrix multiply by W1, add b1, apply ReLU  (3→64)
  3. Matrix multiply by W2, add b2, apply ReLU  (64→32)
  4. Matrix multiply by W3, add b3, apply ReLU  (32→16)
  5. Matrix multiply by W4, add b4, no activation (16→1)
  6. Output = predicted house price in $k

Spring Boot integration:
  Easiest  → Call Python FastAPI over HTTP (REST API approach)
  Better   → Convert to ONNX, run inside JVM (no Python dependency)
  Advanced → Use TensorFlow Java with SavedModel format
```
