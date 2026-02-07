import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# =========================
# CONFIGURATION
# =========================
IMG_SIZE = (100, 100)
N_PCA_COMPONENTS = 100
TEST_SIZE = 0.3
RANDOM_STATE = 42

BASE_DIR = os.getcwd()   # Works in Jupyter & VS Code
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "faces")

# =========================
# LOAD DATASET
# =========================
def load_dataset():
    X, y = [], []
    class_names = []
    label = 0

    print("Dataset path:", DATASET_PATH)
    print("Path exists:", os.path.exists(DATASET_PATH))

    for person in os.listdir(DATASET_PATH):
        person_path = os.path.join(DATASET_PATH, person)
        if not os.path.isdir(person_path):
            continue

        print("Reading person:", person)
        class_names.append(person)

        for img_name in os.listdir(person_path):
            if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, IMG_SIZE)
            X.append(img.flatten())
            y.append(label)

        label += 1

    return np.array(X), np.array(y), class_names


# =========================
# MAIN
# =========================
def main():
    print("\nLoading dataset...\n")
    X, y, class_names = load_dataset()

    print("\nDataset shape:", X.shape)
    print("Number of persons:", len(class_names))

    # Normalize
    X = X / 255.0

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # =========================
    # PCA
    # =========================
    print("\nApplying PCA...\n")
    pca = PCA(n_components=N_PCA_COMPONENTS, whiten=True, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    # =========================
    # SHOW EIGENFACES
    # =========================
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.ravel()

    for i in range(10):
        eigenface = pca.components_[i].reshape(IMG_SIZE)
        axes[i].imshow(eigenface, cmap="gray")
        axes[i].set_title(f"Eigenface {i}")
        axes[i].axis("off")

    plt.suptitle("Top Eigenfaces (PCA Output)")
    plt.show()

    # =========================
    # ANN (MLP Classifier)
    # =========================
    print("\nTraining ANN...\n")
    clf = MLPClassifier(
        hidden_layer_sizes=(150, 100),
        activation="relu",
        solver="adam",
        max_iter=800,
        random_state=RANDOM_STATE
    )

    clf.fit(X_train_pca, y_train)

    y_pred = clf.predict(X_test_pca)

    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f"\nFinal Accuracy: {accuracy:.2f}%")

    # =========================
    # SHOW PREDICTIONS AS IMAGES
    # =========================
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()

    for i in range(12):
        img = X_test[i].reshape(IMG_SIZE)
        pred_name = class_names[y_pred[i]]
        true_name = class_names[y_test[i]]

        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(f"Pred: {pred_name}\nTrue: {true_name}")
        axes[i].axis("off")

    plt.suptitle("Face Recognition Results (ANN Predictions)")
    plt.show()


# =========================
# RUN
# =========================
if __name__ == "__main__":
    main()
