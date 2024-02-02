# Shopping

This Python script is designed to predict whether a user will make a purchase on a shopping website based on their browsing behavior. It utilizes a K-Nearest Neighbors (KNN) classifier to make these predictions, evaluating the model's performance by calculating its sensitivity (true positive rate) and specificity (true negative rate).

## Requirements

- Python 3
- scikit-learn

To install scikit-learn, run: `pip install scikit-learn`

## Dataset

The script expects a CSV file containing the dataset with the following columns:

1. Administrative: Number of visits to administrative pages.
2. Administrative_Duration: Total time spent on administrative pages.
3. Informational: Number of visits to informational pages.
4. Informational_Duration: Total time spent on informational pages.
5. ProductRelated: Number of visits to product-related pages.
6. ProductRelated_Duration: Total time spent on product-related pages.
7. BounceRates: The bounce rate of the website.
8. ExitRates: The exit rate of the website.
9. PageValues: The page value of the website.
10. SpecialDay: Closeness to a special day.
11. Month: Month of the visit.
12. OperatingSystems: Operating system of the user.
13. Browser: Browser used by the user.
14. Region: Region of the user.
15. TrafficType: Traffic type.
16. VisitorType: Type of visitor (Returning or New).
17. Weekend: Indicates whether the visit occurred on a weekend.
18. Label: Indicates whether a purchase was made (TRUE or FALSE).

## Usage

To run the script, use the following command: `python shopping.py <path_to_csv_file>`

Replace `<path_to_csv_file>` with the path to your CSV file containing the dataset.

## Features

- **Data Loading and Preprocessing**: Converts categorical data into numerical representations suitable for the KNN algorithm.
- **Model Training**: Trains a K-Nearest Neighbors classifier with the provided data.
- **Prediction**: Predicts whether a user will make a purchase based on their website browsing behavior.
- **Evaluation**: Calculates and prints the model's sensitivity (true positive rate) and specificity (true negative rate).

## Evaluation Metrics

- **Sensitivity (True Positive Rate)**: The proportion of actual positive cases that were correctly identified.
- **Specificity (True Negative Rate)**: The proportion of actual negative cases that were correctly identified.