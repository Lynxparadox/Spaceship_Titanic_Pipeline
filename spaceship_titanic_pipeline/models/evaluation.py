
from sklearn.metrics import accuracy_score, classification_report

from train import train_model

def evaluate():

    model, X_test, y_test = train_model()

    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    print("Model evaluation completed.")
    print("Accuracy:", acc)
    print(classification_report(y_test, y_pred))

    return acc

if __name__ == "__main__":
    evaluate()
