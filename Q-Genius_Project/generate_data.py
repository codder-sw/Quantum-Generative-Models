import numpy as np
import matplotlib.pyplot as plt
from train_quantum_model import vqc, train_data, train_labels

# 1. मॉडल से प्रेडिक्शन लेना
print("--- Step 4: Generating Predictions from Quantum Model ---")
predictions = vqc.predict(train_data[:5]) # Pehle 5 samples test karte hain

# 2. Result Comparison
print(f"Original Labels:  {train_labels[:5]}")
print(f"Quantum Predicted: {predictions}")

# 3. Visualization of Success
plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(train_data[i].reshape(4, 4), cmap='gray') # 16 features -> 4x4 grid
    plt.title(f"Pred: {predictions[i]}")
    plt.axis('off')
plt.show()

print("✅ SUCCESS: Quantum Inference complete!")