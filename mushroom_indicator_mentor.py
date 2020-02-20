"""
    Mushroom Indicator Mentor

    Author:
    - Samuel Menzel @ www.github.com/sjmenzel
"""

import pandas as pd

TRAINING_DATASET = 'mushroom_training_dataset.csv'

TRAINED_FILE_NAME = 'mushroom_indicator_trained.py'

TARGET_VARIABLE = 'mushroom'        # mushroom == 1, means mushroom is poisonous


def find_best_attr(df, tv):
    best_attr, best_coeff = None, 0

    for attr, row in df.iterrows():
        coeff = round(row[tv], 3)   # Rounded to 3 decimals

        # Ideally -1.0 or +1.0 coefficient
        if abs(coeff) > best_coeff and attr != tv:
            best_attr, best_coeff = attr, coeff

    return best_attr, best_coeff


def main():
    # Read csv file
    df = pd.read_csv(TRAINING_DATASET)

    # Use the pearson method to calculate cross correlation coefficients for attributes
    df = df.corr(method='pearson')

    tv = TARGET_VARIABLE

    # Find the attribute that has the optimal coefficients in relation to the target variable
    best_attr, best_coeff = find_best_attr(df, tv)

    # Print results
    print('The attribute with the best coefficient is {} with value of {}.'.format(best_attr, best_coeff))


if __name__ == '__main__':
    main()
