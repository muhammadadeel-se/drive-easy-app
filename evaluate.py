from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
from app import classify_email_with_gpt2


def evaluate_model():
  print("Loading dataset for evaluation...")
  try:
    df = pd.read_csv("emails.csv")
  except FileNotFoundError:
    print(
        "Error: 'emails.csv' not found. Please ensure your dataset is in the"
        " folder."
    )
    return

  if "subject" not in df.columns or "category" not in df.columns:
    print("Dataset missing required 'subject' or 'category' columns.")
    return

  X = df[["subject", "body"]].fillna("")
  y_true = df["category"]

  # 1. Split CSV into train/test using sklearn (80% train, 20% test)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y_true, test_size=0.2, random_state=42
  )

  print(
      f"Dataset split: {len(X_train)} training samples, {len(X_test)} test"
      " samples."
  )

  y_pred = []

  # 2. Run classification on test set with confidence threshold fallback
  print("Running classification on test set...")
  for _, row in X_test.iterrows():
    category, confidence = classify_email_with_gpt2(row["subject"], row["body"])

    # 5. Apply threshold to flag uncertain emails (< 0.6)
    if confidence < 0.6:
      category = "Uncertain"

    y_pred.append(category)

  # 3. Compute accuracy and confusion matrix
  accuracy = accuracy_score(y_test, y_pred)
  conf_matrix = confusion_matrix(y_test, y_pred, labels=y_test.unique())

  print("\n=== Evaluation Results ===")
  print(f"Overall Accuracy: {accuracy:.4f} (Requirement: > 0.50)")
  print("\nConfusion Matrix:")
  print(conf_matrix)


if __name__ == "__main__":
  evaluate_model()