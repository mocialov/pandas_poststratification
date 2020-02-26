import pandas as pd
import numpy as np
import sys

def get_means(data, columns_of_interest, old_column, new_column):
    """Makes copy of the dataframe, calculates mean of every group in a copied dataframe, changes the column nam
e of the copied dataframe, and returns the new dataframe

    Keyword arguments:
    data -- dataframe
    columns_of_interest -- columns to group the data by
    old_column -- old column name
    new_column -- new column name
    """

    try:
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert isinstance(columns_of_interest, list)
        assert isinstance(old_column, str)
        assert isinstance(new_column, str)
    except:
        print ("get_means: Data types are wrong")
        exit(1)
    try:
        assert [column in columns_of_interest for column in list(data.columns)].count(True) == len(columns_of_in
terest)
        assert old_column in list(data.columns)
    except AssertionError:
        print ("get_means: Column names are not found in the dataframes provided")
        exit(1)

    new_data = data.copy()
    new_data = new_data.rename(columns={old_column: new_column}).groupby(columns_of_interest, sort=False, as_ind
ex=False)[new_column].mean()
    return new_data

def get_sums(data, columns_of_interest, old_column, new_column):
    """Makes copy of the dataframe, calculates sums of every group in a copied dataframe, changes the column nam
e of the copied dataframe, and returns the new dataframe

    Keyword arguments:
    data -- dataframe
    columns_of_interest -- columns to group the data by
    old_column -- old column name
    new_column -- new column name
    """

    try:
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert isinstance(columns_of_interest, list)
        assert isinstance(old_column, str)
        assert isinstance(new_column, str)
    except:
        print ("get_sums: Data types are wrong")
        exit(1)
    try:
        assert [column in columns_of_interest for column in data.columns].count(True) == len(columns_of_interest
)
        assert old_column in data.columns
    except AssertionError:
        print ("get_sums: Column names are not found in the dataframes provided")
        exit(1)

    new_data = data.copy()
    new_data = new_data.rename(columns={old_column: new_column}).groupby(columns_of_interest, sort=False, as_ind
ex=False)[new_column].sum()
    return new_data

def divide_columns(data1, column1, data2, column2, by_group, new_column):
    """Makes copy of the dataframe, joins dataframes by their groups and divides column-wise

    Keyword arguments:
    data1 -- dataframe 1
    column1 -- column name from the data1
    data2 -- dataframe 2
    column2 -- column name from the data2
    by_group -- list of columns to group both dataframes by
    new_column -- name of the new column to store the result of the division of the data from column1 and column
2
    """

    try:
        assert data1 is not None
        assert data2 is not None
        assert isinstance(data1, pd.DataFrame)
        assert isinstance(data2, pd.DataFrame)
        assert isinstance(column1, str)
        assert isinstance(column2, str)
        assert isinstance(by_group, list)
        assert isinstance(new_column, str)
    except:
        print ("get_means: Data types are wrong")
        exit(1)
    try:
        assert column1 in list(data1.columns)
        assert column2 in list(data2.columns)
        assert [column in by_group  for column in list(data1.columns)].count(True) == len(by_group)
        assert [column in by_group  for column in list(data2.columns)].count(True) == len(by_group)
    except AssertionError:
        print ("divide_columns: Column name not found in the dataframe provided")
        exit(1)

    new_data = data1.copy()
    new_data = new_data.join(data2.set_index(by_group), on=by_group, lsuffix='_left', rsuffix='_right')
    new_data[new_column] = new_data[column1] / new_data[column2]
    return new_data

def min_max_normalise(data, column, new_column, minimum=0.5, maximum=1.5):
    """Makes copy of the dataframe and normalises it in the provided bounds

    Keyword arguments:
    data -- dataframe
    column -- column name to normalise the data
    new_column -- name of the new column to store the results of the normalisation
    minimum -- minimum bound for the normalisation (default: 0.5)
    maximum -- maximum bound for the normalisation (default: 1.5)
    """

    try:
        assert data is not None
        assert isinstance(data, pd.DataFrame)
        assert isinstance(column, str)
        assert isinstance(new_column, str)
        assert isinstance(minimum, float)
        assert isinstance(maximum, float)
    except:
        print ("min_max_normalise: Data types are wrong")
        exit(1)
    try:
        assert column in list(data.columns)
    except AssertionError:
        print ("min_max_normalise: Column name not found in the dataframe provided")
        exit(1)

    new_data = data.copy()
    new_data[new_column] = (maximum - minimum) * (new_data[column] - new_data[column].min()) / (new_data[column]
.max() - new_data[column].min()) + minimum
    return new_data

def multiply_columns(data1, column1, data2, column2, by_group, new_column):
    """Makes copy of the dataframe1 and multiplies two columns from two different dataframes

    Keyword arguments:
    data1 -- dataframe 1
    column1 -- column name from the data1
    data2 -- dataframe 2
    column2 -- column from the data2
    by_group -- list of columns to group both dataframes by
    new_column -- name of the new column to store the results of the multiplication of the data from column1 and
 column2
    """

    try:
        assert data1 is not None
        assert data2 is not None
        assert isinstance(data1, pd.DataFrame)
        assert isinstance(data2, pd.DataFrame)
        assert isinstance(column1, str)
        assert isinstance(column2, str)
        assert isinstance(by_group, list)
        assert isinstance(new_column, str)
    except:
        print ("multiply_columns: Data types are wrong")
        exit(1)
    try:
        assert column1 in list(data1.columns)
        assert column2 in list(data2.columns)
        assert [column in by_group  for column in list(data1.columns)].count(True) == len(by_group)
        assert [column in by_group  for column in list(data2.columns)].count(True) == len(by_group)
    except AssertionError:
        print ("multiply_columns: Column name not found in the dataframe provided")
        exit(1)

    new_data = data1.copy()
    new_data = new_data.join(data2.set_index(by_group), on=by_group, lsuffix='_left', rsuffix='_right')
    new_data[new_column] = new_data[column1] * new_data[column2]
    return new_data

def test_poststratification(total_population_percentage, poststratified_predictions, group_dataframes_by, mappin
g):
    """Test randomly sampled 5 cases of postratification
    """

    try:
        assert total_population_percentage is not None
        assert isinstance(total_population_percentage, pd.DataFrame)
        assert poststratified_predictions is not None
        assert isinstance(poststratified_predictions, pd.DataFrame)
        assert isinstance(group_dataframes_by, list)
        assert isinstance(mapping, tuple)
        assert len(mapping) == 3
    except:
        print ("test_poststratification: Data types are wrong")
        exit(1)
    try:
        assert mapping[0] in list(poststratified_predictions.columns)
        assert mapping[1] in list(total_population_percentage.columns)
        assert mapping[2] in list(poststratified_predictions)
    except AssertionError:
        print ("test_poststratification: Column name not found in the dataframe provided")
        exit(1)

    random_samples = total_population_percentage.sample(n=5)
    for idx, item2 in random_samples.iterrows():
        for idx2, item in poststratified_predictions.iterrows():
            if all([item2[grouping_item] == item[grouping_item] for grouping_item in group_dataframes_by]):
                if item[mapping[0]] * item2[mapping[1]] != item[mapping[2]]:
                    return False
                break
    return True

def main(file1_path, file2_path):
    """Performing stratification (aka) weighting of the data from two different datasets"""
    group_dataframes_by = ["nuts1", "gender", "education"]

    loaded_population_data = pd.read_csv(file1_path)
    population_means = get_means(loaded_population_data, group_dataframes_by, "population", "population_mean")
    population_sums = get_sums(loaded_population_data, ["nuts1"], "population", "population_sum")
    total_population_percentage = divide_columns(population_means, "population_mean", population_sums, "populati
on_sum", ["nuts1"], "population_percentage")
    min_max_normalised_total_population_percentage = min_max_normalise(total_population_percentage, "population_
percentage", "population_weight", 0.5, 1.5)
    loaded_prediction_data = pd.read_csv(file2_path)
    poststratified_predictions = multiply_columns(loaded_prediction_data, "predicted_probability", min_max_norma
lised_total_population_percentage, "population_weight", group_dataframes_by, 'predicted_probability_poststratifi
ed')

    try:
        assert test_poststratification(min_max_normalised_total_population_percentage, poststratified_prediction
s, group_dataframes_by, ("predicted_probability", "population_weight", "predicted_probability_poststratified"))
    except AssertionError:
        print ("main: Testing of the poststratification results failed")

    poststratified_predictions.to_csv("predicted_probability_poststratified.csv")
    print ("Poststratified file was saved")


if __name__ == '__main__':
    try:
        assert len(sys.argv) == 3

        file1_path = sys.argv[1]
        file2_path = sys.argv[2]

        main(file1_path, file2_path)

    except AssertionError:
        print ("Two inputs are required. File 1 (predictions) and File 2 (census)")
