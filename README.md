
## 🧠 EEG-Depression-Detection

A deep learning project for **EEG-based depression detection** using a hybrid **CNN + BiLSTM architecture**. This project demonstrates a full machine learning pipeline from preprocessing to deployment.

---

## 🚀 Highlights

* 🔬 EEG signal classification (HC / DEAP / MDD)
* 🧠 Hybrid **CNN + BiLSTM** model
* 🔁 K-Fold Cross Validation
* 📊 Multiple evaluation metrics (Precision, Recall, F1-score)
* 📉 Training visualization (Accuracy & Loss curves)
* 🌐 Interactive **Streamlit demo**

---

## 📁 Project Structure

```
EEG-Depression-Detection/
│
├── data/                  # Data description (no raw data)
├── preprocess/           # Data preprocessing
├── models/               # Model architecture
├── train.py              # Training pipeline
├── evaluate.py           # Evaluation & metrics
├── app.py                # Streamlit demo
│
├── results/              # Output figures & models
│
├── requirements.txt
└── README.md
```

---

## 🧠 Model Architecture

### 🔹 Overview

The model combines:

* **CNN** → Extract spatial features from EEG signals
* **BiLSTM** → Capture temporal dependencies

---

### 📊 Architecture Diagram (Conceptual)

```
Input (8000 × 15)
        │
        ▼
[ Conv1D (64) + BN + MaxPool ]
        │
        ▼
[ Conv1D (128) + BN + MaxPool ]
        │
        ▼
[ Conv1D (256) + BN + MaxPool ]
        │
        ▼
      Dropout
        │
        ▼
[ BiLSTM (128) ]
        │
        ▼
[ BiLSTM (64) ]
        │
        ▼
[ Dense (128) ]
        │
        ▼
[ Dense (64) ]
        │
        ▼
[ Softmax (3 classes) ]
        │
        ▼
Output: HC / DEAP / MDD
```

---

## ⚙️ Methodology

### 1. Data Preprocessing

* Shape normalization: `(15, 8000) → (8000, 15)`
* Standardization using `StandardScaler`
* Consistent scaling across folds

### 2. Training Strategy

* **K-Fold Cross Validation (k=5)**
* Early Stopping
* Learning Rate Scheduling
* Model Checkpointing

### 3. Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-score

---

## 📈 Results

### Training Curves

* Accuracy vs Epoch
* Loss vs Epoch

### Confusion Matrix

Provides detailed classification performance across all classes.

---

## 🌐 Demo (Streamlit)

Run the interactive demo:

```
streamlit run app.py
```

Features:

* Upload EEG `.npy` file
* Real-time prediction
* Output class label

---

## 🛠 Installation

```
pip install -r requirements.txt
```

---

## ▶️ Usage

### Train Model

```
python train.py
```

### Evaluate Model

```
python evaluate.py
```

---

## 📌 Future Improvements

* EEG data augmentation
* Transformer-based models
* Real-time EEG streaming support
* Web deployment (Docker + Cloud)

---

## 💡 Author

Vincent Liu

