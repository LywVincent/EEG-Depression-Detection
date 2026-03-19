import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import joblib
import gc

from preprocess.preprocess import load_data, preprocess_data
from models.model import build_complex_model

# 数据路径（改成相对路径更专业）
data_root = "data/"
hc_path = data_root + "HC"
deap_path = data_root + "DEAP"
mdd_path = data_root + "MDD"


def kfold_cross_validation(data, labels, k=5):
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kfold.split(data)):
        print(f"\nFold {fold+1}")

        X_train, X_val = data[train_idx], data[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        X_train, scaler = preprocess_data(X_train)
        X_val, _ = preprocess_data(X_val, scaler)

        joblib.dump(scaler, f"results/scaler_fold_{fold+1}.pkl")

        model = build_complex_model(X_train.shape[1:])

        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ModelCheckpoint(f"results/best_model_fold_{fold+1}.h5",
                            save_best_only=True),
            ReduceLROnPlateau(patience=5)
        ]

        model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks
        )

        y_pred = model.predict(X_val)
        y_pred = np.argmax(y_pred, axis=1)

        print("Precision:", precision_score(y_val, y_pred, average='weighted'))
        print("Recall:", recall_score(y_val, y_pred, average='weighted'))
        print("F1:", f1_score(y_val, y_pred, average='weighted'))

        del model
        gc.collect()


if __name__ == "__main__":
    tf.keras.backend.clear_session()
    data, labels = load_data(hc_path, deap_path, mdd_path)
    kfold_cross_validation(data, labels)
