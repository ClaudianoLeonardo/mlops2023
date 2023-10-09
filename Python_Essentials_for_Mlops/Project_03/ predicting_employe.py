import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

def load_and_clean_data(file_path):
    """
    Load the dataset and perform data cleaning.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    df = pd.read_csv(file_path)

    # Clean up department column
    df["department"] = df["department"].str.strip()

    # Drop unnecessary columns
    df.drop(["date", "idle_time", "idle_men", "wip", "no_of_style_change"], axis=1, inplace=True)

    # Convert quarter to integers
    quarter_mapping = {"Quarter1": 1, "Quarter2": 2, "Quarter3": 3, "Quarter4": 4, "Quarter5": 4}
    df["quarter"] = df["quarter"].map(quarter_mapping).astype(int)

    # Convert data types
    df["no_of_workers"] = df["no_of_workers"].astype(int)
    df["actual_productivity"] = df["actual_productivity"].round(2)

    # Rename columns
    df.rename(columns={"department": "dept_sweing"}, inplace=True)

    # One-hot encode categorical variables
    df = pd.get_dummies(df, columns=["quarter", "day", "team"], prefix=["q", None, "team"]).drop(["quarter", "day", "team"], axis=1)

    # Map department to integers
    df["dept_sweing"] = df["dept_sweing"].map({"finishing": 0, "sweing": 1})

    return df

def train_decision_tree_classifier(df, max_depth=3, random_state=24):
    """
    Train a Decision Tree Classifier.

    Args:
        df (pd.DataFrame): DataFrame with features and target.
        max_depth (int): Maximum depth of the decision tree.
        random_state (int): Random seed for reproducibility.

    Returns:
        DecisionTreeClassifier: Trained decision tree classifier.
    """
    X = df.drop(["actual_productivity", "productive"], axis=1)
    y = df["productive"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=random_state)

    tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    tree.fit(X_train, y_train)

    return tree, X_test, y_test

def evaluate_decision_tree_classifier(tree, X_test, y_test):
    """
    Evaluate a Decision Tree Classifier.

    Args:
        tree (DecisionTreeClassifier): Trained decision tree classifier.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True labels for the test set.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    y_pred = tree.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    evaluation_metrics = {
        "Accuracy": round(accuracy, 2),
        "Confusion Matrix": conf_matrix,
        "Precision": round(precision, 2),
        "Recall": round(recall, 2),
        "F1 Score": round(f1, 2),
    }

    return evaluation_metrics

def plot_decision_tree(tree, feature_names):
    """
    Plot the Decision Tree.

    Args:
        tree (DecisionTreeClassifier): Trained decision tree classifier.
        feature_names (list): List of feature names.
    """
    plt.figure(figsize=[20.0, 8.0])
    _ = plot_tree(tree, feature_names=feature_names, class_names=["Unproductive", "Productive"], filled=True, rounded=False, proportion=True, fontsize=11)

def train_random_forest_classifier(df, random_state=24):
    """
    Train a Random Forest Classifier.

    Args:
        df (pd.DataFrame): DataFrame with features and target.
        random_state (int): Random seed for reproducibility.

    Returns:
        RandomForestClassifier: Trained random forest classifier.
    """
    X = df.drop(["actual_productivity", "productive"], axis=1)
    y = df["productive"]

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=True, random_state=random_state)

    forest = RandomForestClassifier(oob_score=True, random_state=random_state)
    forest.fit(X_train, y_train)

    return forest

def main():
    # Load and clean the data
    df = load_and_clean_data("garments_worker_productivity.csv")

    # Train a Decision Tree Classifier
    tree, X_test, y_test = train_decision_tree_classifier(df)

    # Evaluate the Decision Tree Classifier
    decision_tree_metrics = evaluate_decision_tree_classifier(tree, X_test, y_test)
    print("Decision Tree Classifier Metrics:")
    for key, value in decision_tree_metrics.items():
        print(f"{key}: {value}")

    # Plot the Decision Tree
    plot_decision_tree(tree, df.columns)

    # Train a Random Forest Classifier
    forest = train_random_forest_classifier(df)

    # Predict using the Random Forest
    y_pred_forest = forest.predict(X_test)

    # Calculate accuracy for the Random Forest
    accuracy_forest = accuracy_score(y_test, y_pred_forest)
    print("\nRandom Forest Classifier Metrics:")
    print("Accuracy (Random Forest):", round(accuracy_forest, 2))
    print("Out Of Bag Score:", round(forest.oob_score_, 2))

if __name__ == "__main__":
    main()