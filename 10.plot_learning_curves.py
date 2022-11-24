import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.linear_model import Lasso, Ridge, LinearRegression, ElasticNet
import utils
from utils import plot_multi_learning_curves

np.random.seed(42)
#dataset = pd.read_csv('./datasets/3_512_x_main.csv')
#target = pd.read_csv('./datasets/3_512_y_main.csv')
#####

df = pd.read_csv("data/df_out1.csv")
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
df = df.iloc[:2000,:] #select training examples
x_values = np.array(df.drop('logS_aq_avg', axis = 1))
labels = df['logS_aq_avg']
y_values = np.array(labels)

# estimator2 = SVR()
estimator1 = RandomForestRegressor(n_jobs=-1)
# estimator3 = DecisionTreeRegressor()

plot_multi_learning_curves(x_values, y_values, estimator1,
                           random_seed = 42, testsize = 0.2, mode = 'r2', autosave = 'y', interval = None)
