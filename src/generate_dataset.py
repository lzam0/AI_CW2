import pandas as pd
import numpy as np

# Define parameters
num_samples_per_class = 20  # Total 200 samples for letters A-J
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
num_features = 63  # 21 landmarks * (x, y, z)

data = []

for label in labels:
    for i in range(num_samples_per_class):
        # Create a unique instance ID
        instance_id = f"dummy_{label}_{i}"
        
        # Generate 63 random float values for coordinates
        features = np.random.rand(num_features).tolist()
        
        # Combine ID, features, and label
        row = [instance_id] + features + [label]
        data.append(row)

# Create column names: instance_id, x0, y0, z0, ..., x20, y20, z20, label
columns = ['instance_id']
for i in range(21):
    columns.extend([f'x{i}', f'y{i}', f'z{i}'])
columns.append('label')

# Convert to DataFrame
df_dummy = pd.DataFrame(data, columns=columns)

# Save to CSV as recommended in the spec
df_dummy.to_csv('dummy_asl_features.csv', index=False)

# This file generates a dummy dataset for ASL hand landmarks
# This dataset is NOT cleaned and will need to be preprocessed
