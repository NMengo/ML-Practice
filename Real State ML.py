import os
import tarfile
import urllib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from scipy import stats
from sklearn.model_selection import (train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV,
                                     RandomizedSearchCV)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """
    Creates a datasets/housing directory in
    your workspace, downloads the housing.tgz file, and extracts the housing.csv file from
    it in this directory.
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)

fetch_housing_data()
housing_data = load_housing_data()

# Data exploration
print(housing_data.info())

# How many categorical values are and how many in each?
print(housing_data['ocean_proximity'].value_counts())

# Stat info
print(housing_data.describe())

# Glance of distributions
housing_data.hist(bins=50, figsize=(20,15))

# Split dataset  into train/split
# train_set, test_set = train_test_split(housing_data, test_size=0.2, random_state=42)
"""
If the dataset is large enough (especially relative to number of attributes), this case would be fine.
But if not, you could get into a significant sampling bias.

In other words, when you choose 20% of your dataset as test set, you need to ensure that the sample is representative
of the whole set, by Stratified Sampling.

In this case, median income seems to be an importan attribute as a predictor. In this sense, it would be desirable that 
the test set is representative of the various categories of incomes.

Since this is a continuous numerical attribute, it should be converted into clusters.
"""
# Creating income clusters
housing_data['income_cat'] = pd.cut(housing_data['median_income'], bins=[0., 1.5, 3.0, 4.5, 6, np.inf], labels=
                                    [1, 2, 3, 4, 5])
housing_data['income_cat'].hist()
# plt.show()

# Stratified aproach
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing_data, housing_data['income_cat']):
    strat_train_set = housing_data.loc[train_index]
    strat_test_set = housing_data.loc[test_index]

# Random aproach
train_set, test_set = train_test_split(housing_data, test_size=0.2, random_state=42)

# Checking proportions
comp = pd.DataFrame({
    'Overall': housing_data['income_cat'].value_counts() / len(housing_data),
    'Stratified': strat_test_set['income_cat'].value_counts() / len(strat_test_set),
    'Random': test_set['income_cat'].value_counts() / len(test_set)
}).sort_index()
comp['Rand %error'] = ((comp['Random'] - comp['Overall']) / comp['Overall']) * 100
comp['Strat %error'] = ((comp['Stratified'] - comp['Overall']) / comp['Overall']) * 100
print('-----------------------------------------------------------------------')
print(comp)

# Removing Clustering to leave data as it originally was.
strat_train_set = strat_train_set.drop('income_cat', axis=1)
strat_test_set = strat_test_set.drop('income_cat', axis=1)

# ======================================================================================================================
# EDA
# Making sure just to work with train set.
# Moreover, creating a copy in order to manipulate it freely without risk of harming original data.
housing = strat_train_set.copy()

housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4, s=housing['population']/100,
             c=housing['median_house_value'], cmap=plt.get_cmap('jet'), colorbar=True, label='population', figsize=(10,7))
# By setting opacity, it's easier to spot high density data points.
# radius (s) of each circle represents district's population
# color (c) represents price
# color map (cmap='jet') ranges from blue (low) to red (high prices).

# Standard Correlation Coefficient between every pair of attributes - Pearson's r
corr_matrix = housing.corr()
print('-----------------------------------------------------------------------')
print(corr_matrix['median_house_value'].sort_values(ascending=False))

# scatter_matrix() from pandas plots every numerical attribute against every other numerical attribute, and it's histogram.
focus_attributes = ["median_house_value", "median_income", "total_rooms",
"housing_median_age"]
scatter_matrix(housing[focus_attributes], figsize=(12, 8))
# Here is easy to grasp again that the most promising attribute to predict median house value, is median income.


housing = strat_train_set.drop('median_house_value', axis=1)
housing_labels = strat_train_set['median_house_value'].copy()

# ======================================================================================================================
# Data Engineering

# ===============================
# Nans treatment.
imputer = SimpleImputer(strategy= 'median')
# Non numerical attributes must be dropped, obviously, to calculate and replace nan for the median.
# We could apply it only to the attribute with missing values, but we don't know if new info will have nan's in
# other attributes too.
housing_num = housing.drop('ocean_proximity', axis=1)

# This computes the median of each attribute and stores it in its statistics_instance variable.
imputer.fit(housing_num)

# Now we can apply this "trained" imputer to transform the training set by replacing missing values.
X = imputer.transform(housing_num)
# This still is an array, so we put it back into df.
housing_tr = pd.DataFrame(X, columns= housing_num.columns, index=housing_num.index)

# ===============================
# Handling text as numerical categorical attributes (if possible).

"""
The problem with the following approach is that ML algorithms will assume that two nearby values are more similiar 
than two distant values. Which may be the case sometimes (as 1:'bad' / 5:'excelent').
"""
ordinal_encoder = OrdinalEncoder()
ocean_proximity_encoded = ordinal_encoder.fit_transform(housing['ocean_proximity'].values.reshape(-1,1))

print('-----------------------------------------------------------------------')
print(ordinal_encoder.categories_) # List of encoded categories.


"""
To solve this, we can use One-hot encoding, that will create one binary attribute per category.
The output will be an sparce matrix since the original one would be full of 0's and a single 1 per row, wasting a lot
of memory.

For converting it to an array, you can simply use .toarray().
"""
cat_encoder = OneHotEncoder()
ocean_proximity_1hot = cat_encoder.fit_transform(housing['ocean_proximity'].values.reshape(-1,1))

# ===============================
# Feature Scaling

"""
ML algorithms don't perform well when input attributes have very different scales.
There are two common ways to get attributes to have same scale:
                * Min-max Scaling
                * Standarization

    +Min-max scaling (many people call this normalization) is the simplest: values are shifted
     and rescaled so that they end up ranging from 0 to 1. - MinMaxScaler
     
    +Standardization first it subtracts the mean value (so standardized values always have a zero mean), and then it 
     divides by the standard deviation so that the resulting distribution has unit variance. 
     Unlike min-max scaling, standardization does not bound values to a specific range. - StandardScaler
     
Also, suppose there is a mistake in one value of an attribute, making it to be very far away from the rest.
Min-max scaling would crush all other values very close to one extreme (1 or 0).
"""

# ===============================
# Custom Transformers

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
# Combines two attributes to get more useful info.
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
        return np.c_[X, rooms_per_household, population_per_household,
            bedrooms_per_room]
# ===============================
# Transformation Pipelines

"""
There are many data transformation steps that need to be executed in the right order. Fortunately, Scikit-Learn provides
the Pipeline class to help with such sequences of transformations.

When you call the pipeline’s fit() method, it calls fit_transform() sequentially on
all transformers, passing the output of each call as the parameter to the next call until
it reaches the final estimator, for which it calls the fit() method.
"""

num_pipeline = Pipeline([('imputer', SimpleImputer(strategy="median")),
                         ('attribs_adder', CombinedAttributesAdder()),
                         ('std_scaler', StandardScaler()),])
# housing_num_tr = num_pipeline.fit_transform(housing_num)

"""
In this case, we only worked the numerical attributes.
In order to work on both, numerical and categorical, in the same pipeline we can use ColumnTransformer.
"""

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)


# ======================================================================================================================
# Training and Evaluation on Training Set
lr1 = LinearRegression()
lr1.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print('-----------------------------------------------------------------------')
print('Predictions:', lr1.predict(some_data_prepared))
print('Labels:', list(some_labels))

housing_predictions = lr1.predict(housing_prepared)
lr1_mse = mean_squared_error(housing_labels, housing_predictions)
lr1_rmse = np.sqrt(lr1_mse)
print('RMSE:',lr1_rmse)

"""
RMSE is quite high in training set.
Clear example of model underfitting training data. Meaning that the features do not provide enough information
to make good predictions or that the model is not the right fit.

In this sense, we will try with a more powerful model, capable of finding complex nonlinear relationships in data.
"""

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions_tree = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions_tree)
tree_rmse = np.sqrt(tree_mse)
print('-----------------------------------------------------------------------')
print('Tree RMSE:', tree_rmse)

"""
RMSE= 0
Something seems to be wrong. More likely that the model has badly overfit the data.
As we can't touch yet test set, we need to look for another way out in order to check what happened.

So we need to use part of the training set for training and part for model validation.
Known as Cross-Validation.

This could be done manually by using train_test_split() to split the training set into a smaller training set and 
a validation set. Then training the models against the smaller training set and evaluating it against the validation set.

Luckily, Scikit-Learn has a feature called K-fold cross-validation.
This will automatically randomly split the training set into (cv) distinct subsets called folds, then it trains the
model (cv) times, picking a different fold for evaluation every time and training on the other 9 folds.

The result is an array containing the 10 evaluation scores.
"""

tree_mse_scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-tree_mse_scores)

"""
Scikit-Learn’s cross-validation features expect a utility function (greater is better) rather than a cost function
(lower is better), so the scoring function is actually the opposite of the MSE (i.e., a negative value), which is why
the preceding code computes -scores before calculating the square root.
"""

def display_scores(scores):
    print('Scores:', scores)
    print('Mean:', scores.mean())
    print('Std dev:', scores.std())

print('-----------------------------------------------------------------------')
display_scores(tree_rmse_scores)

"""
It's even worse than the result of de Linear Regression, due to the extreme overfitting. 
Let's apply also cross-validation to LR1, for double-checking.
"""

lin_mse_scores = cross_val_score(lr1, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-lin_mse_scores)

print('-----------------------------------------------------------------------')
display_scores(lin_rmse_scores)

# Last model try

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions_forest = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)

print('-----------------------------------------------------------------------')
print('RandomForest:', forest_rmse)
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, cv=10, scoring='neg_mean_squared_error')
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)


# ======================================================================================================================
# Tuning the model

# ===============================
# Grid search

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators':[3, 10, 30], 'max_features':[2,4,6,8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap':[False], 'n_estimators':[3, 10], 'max_features': [2,3,4]},
]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring='neg_mean_squared_error', return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)

print('-----------------------------------------------------------------------')
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'], cvres['params']):
    print(np.sqrt(-mean_score), params)

# ===============================
# Randomized Search

"""
When you are exploring more than a few combinations, you may want to use RandomizedSearchCV instead.
Instead of trying all possible combinations, it evaluates a given number of random combinations by selecting
a random value for each hyperparameter at every iteration.
"""

# param_distribs = {
#         'n_estimators': randint(low=1, high=200),
#         'max_features': randint(low=1, high=8),
#     }
#
# forest_reg = RandomForestRegressor(random_state=42)
# rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
#                                 n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
# rnd_search.fit(housing_prepared, housing_labels)
#
# cvres = rnd_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(np.sqrt(-mean_score), params)


# ======================================================================================================================
# Evaluating model on the Test set

final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

print('-----------------------------------------------------------------------')
confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
conf_int = np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
print(final_rmse)
print('95% Confidence Interval:', conf_int)
