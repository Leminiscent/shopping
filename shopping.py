"""
This script aims to predict whether a user will make a purchase on a website based on their behavior. 
It employs a K-Nearest Neighbors classifier to make these predictions. The data used for training and 
testing the model includes various features such as the number of times different types of pages were 
visited, the duration of these visits, bounce rates, and more. The script evaluates the model's 
performance by calculating its sensitivity (true positive rate) and specificity (true negative rate).
"""

import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4  # Proportion of the dataset to include in the test split


def main():
    """
    Main function to execute the program workflow: loading data, splitting into training and testing
    sets, training the model, making predictions, and evaluating the model's performance.
    """
    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python shopping.py data")

    # Load data from spreadsheet and split into train and test sets
    evidence, labels = load_data(sys.argv[1])
    X_train, X_test, y_train, y_test = train_test_split(
        evidence, labels, test_size=TEST_SIZE
    )

    # Train model and make predictions
    model = train_model(X_train, y_train)
    predictions = model.predict(X_test)

    # Evaluate model's performance
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    """
    Load dataset from a CSV file.

    Parameters:
        filename (str): The path to the CSV file.

    Returns:
        tuple: A tuple containing two lists, `evidence` and `labels`. `evidence` contains the features for
        each instance, and `labels` contains the corresponding outcome (1 for purchase made, 0 otherwise).
    """
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row

        evidence = []
        labels = []
        for row in reader:
            # Convert each row into the appropriate data type
            evidence.append(
                [
                    int(row[0]),  # Administrative
                    float(row[1]),  # Administrative_Duration
                    int(row[2]),  # Informational
                    float(row[3]),  # Informational_Duration
                    int(row[4]),  # ProductRelated
                    float(row[5]),  # ProductRelated_Duration
                    float(row[6]),  # BounceRates
                    float(row[7]),  # ExitRates
                    float(row[8]),  # PageValues
                    float(row[9]),  # SpecialDay
                    get_month(row[10]),  # Month
                    int(row[11]),  # OperatingSystems
                    int(row[12]),  # Browser
                    int(row[13]),  # Region
                    int(row[14]),  # TrafficType
                    get_visitor_type(row[15]),  # VisitorType
                    get_weekend(row[16]),  # Weekend
                ]
            )
            labels.append(1 if row[17] == "TRUE" else 0)  # Label
        return evidence, labels


def get_month(month):
    """
    Convert month from string to numerical representation.

    Parameters:
        month (str): The month as a three-letter abbreviation.

    Returns:
        int: The index of the month in a year (0 for January, 11 for December).
    """
    months = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "June",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]
    return months.index(month)


def get_visitor_type(visitor_type):
    """
    Convert visitor type to a binary representation.

    Parameters:
        visitor_type (str): The type of visitor (e.g., "Returning_Visitor").

    Returns:
        int: 1 for returning visitors, 0 otherwise.
    """
    return 1 if visitor_type == "Returning_Visitor" else 0


def get_weekend(weekend):
    """
    Convert weekend indicator to a binary representation.

    Parameters:
        weekend (str): Indicates whether the visit occurred on a weekend ("TRUE" or "FALSE").

    Returns:
        int: 1 if the visit occurred on a weekend, 0 otherwise.
    """
    return 1 if weekend == "TRUE" else 0


def train_model(evidence, labels):
    """
    Train a K-Nearest Neighbors classifier using the provided training data.

    Parameters:
        evidence (list): The features for each instance in the training data.
        labels (list): The corresponding labels for each instance in the training data.

    Returns:
        KNeighborsClassifier: The trained model.
    """
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(evidence, labels)
    return model


def evaluate(labels, predictions):
    """
    Evaluate the model's performance by calculating sensitivity and specificity.

    Parameters:
        labels (list): The true labels for the test data.
        predictions (list): The predicted labels by the model.

    Returns:
        tuple: A tuple containing the sensitivity and specificity of the model.
    """
    true_positives = sum(
        [1 for true, pred in zip(labels, predictions) if true == 1 and pred == 1]
    )
    true_negatives = sum(
        [1 for true, pred in zip(labels, predictions) if true == 0 and pred == 0]
    )

    sensitivity = true_positives / sum(labels)
    specificity = true_negatives / (len(labels) - sum(labels))

    return sensitivity, specificity


if __name__ == "__main__":
    main()
