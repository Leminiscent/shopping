import csv
import sys

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

TEST_SIZE = 0.4


def main():
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
    sensitivity, specificity = evaluate(y_test, predictions)

    # Print results
    print(f"Correct: {(y_test == predictions).sum()}")
    print(f"Incorrect: {(y_test != predictions).sum()}")
    print(f"True Positive Rate: {100 * sensitivity:.2f}%")
    print(f"True Negative Rate: {100 * specificity:.2f}%")


def load_data(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        next(reader)  # Skip the header row

        evidence = []
        labels = []
        for row in reader:
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
            labels.append(int(row[17]))  # Label
        return evidence, labels


def train_model(evidence, labels):
    """
    Given a list of evidence lists and a list of labels, return a
    fitted k-nearest neighbor model (k=1) trained on the data.
    """
    raise NotImplementedError


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return a tuple (sensitivity, specificity).

    Assume each label is either a 1 (positive) or 0 (negative).

    `sensitivity` should be a floating-point value from 0 to 1
    representing the "true positive rate": the proportion of
    actual positive labels that were accurately identified.

    `specificity` should be a floating-point value from 0 to 1
    representing the "true negative rate": the proportion of
    actual negative labels that were accurately identified.
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
