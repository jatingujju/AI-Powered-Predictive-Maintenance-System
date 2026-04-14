import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

def train_model(df):

    X = df[['temperature', 'vibration', 'pressure']]
    y = df['failure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Accuracy: {acc*100:.2f}%")

    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)

    # 🔥 Plot confusion matrix
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.show()

    # Save model
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/model.pkl")

    print("💾 Model Saved!")