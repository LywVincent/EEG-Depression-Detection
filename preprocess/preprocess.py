import numpy as np
import os
from sklearn.preprocessing import StandardScaler

def load_data(hc_path, deap_path, mdd_path):
    data = []
    labels = []
    label_map = {'HC': 0, 'DEAP': 1, 'MDD': 2}

    folders = {'HC': hc_path, 'DEAP': deap_path, 'MDD': mdd_path}

    for label_folder, folder_path in folders.items():
        print(f"\nProcessing folder: {label_folder}")
        file_count = 0

        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            if file_name.endswith('.npy'):
                try:
                    file_data = np.load(file_path)

                    if file_data.shape != (15, 8000):
                        print(f"Warning: {file_path} shape {file_data.shape}, skipped")
                        continue

                    data.append(file_data)
                    labels.append(label_map[label_folder])
                    file_count += 1

                except Exception as e:
                    print(f"Error loading {file_path}: {e}")

        print(f"Loaded {file_count} files from {label_folder}")

    return np.array(data), np.array(labels)


def preprocess_data(data, scaler=None):
    if scaler is None:
        scaler = StandardScaler()

    num_samples = data.shape[0]
    data_reshaped = data.transpose(0, 2, 1).reshape(-1, 15)
    data_scaled = scaler.fit_transform(data_reshaped)

    return data_scaled.reshape(num_samples, 8000, 15), scaler
