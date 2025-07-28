import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

laptop = pd.read_csv('data/laptop_data.csv')

# drop irrelevant columns
laptop.drop(['Unnamed: 0'], axis=1, inplace=True)
# Equal missing rows across features
laptop.dropna(inplace=True)
# reset index after dropping rows
laptop.reset_index(drop=True, inplace=True)

# The price distribution is wide and skewed, with a large standard deviation.
# Using the quartiles (25%, 50%, 75%) as breakpoints for binning the price into categories to maintain balance of laptops prices
laptop['price_cat'] = pd.cut(laptop['Price'],
                        bins = [0, 32000, 52000, 80000, float('inf')],
                        labels = [0, 1, 2, 3]  # Budget, Midrange, Premium, High-end
                        )

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(laptop, laptop['price_cat']):
	laptop_train_set = laptop.loc[train_index]
	laptop_test_set = laptop.loc[test_index]
	
# drop price_cat. It is not part of the dataset
for set in (laptop_train_set, laptop_test_set):
	set.drop('price_cat', axis=1, inplace=True)

laptop_train_set.to_csv('data/laptop_train_set.csv', index=False)
laptop_test_set.to_csv('data/laptop_test_set.csv', index=False)