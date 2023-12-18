import pandas as pd;

pd.set_option('display.max_columns', 100)
import numpy as np

import warnings

warnings.filterwarnings('ignore')

from tqdm.notebook import tqdm

import re

from functools import partial
from scipy.stats import mode

import matplotlib.pyplot as plt;

plt.style.use('ggplot')
import seaborn as sns
import plotly.express as px

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, FunctionTransformer, PowerTransformer, \
    PolynomialFeatures
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import KNNImputer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV, RepeatedStratifiedKFold, \
    cross_val_score, cross_val_predict
from sklearn.metrics import roc_auc_score, roc_curve, RocCurveDisplay, cohen_kappa_score, log_loss, f1_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_selection import RFE, RFECV
from sklearn.isotonic import IsotonicRegression
from sklearn.calibration import CalibrationDisplay, CalibratedClassifierCV
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from collections import Counter
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, HistGradientBoostingClassifier, \
    GradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import optuna

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')
submission = pd.read_csv('./sample_submission.csv')

print('The dimension of the train dataset is:', train.shape)
print('The dimension of the test dataset is:', test.shape)

print(train.describe())

print(test.describe())

train['Status'].value_counts(normalize=True).plot(kind='bar', color=['steelblue', 'orange', 'green'])
plt.ylabel('Percentage')
# plt.show()

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

sns.boxplot(ax=axes[0], data=train, x='Status', y='Age', hue='Status');
sns.boxplot(ax=axes[1], data=train, x='Status', y='Bilirubin', hue='Status');
sns.boxplot(ax=axes[2], data=train, x='Status', y='Cholesterol', hue='Status');

# plt.show()

fig, axes = plt.subplots(1, 3, figsize=(20, 7))

sns.boxplot(ax=axes[0], data=train, x='Status', y='Albumin', hue='Status');
sns.boxplot(ax=axes[1], data=train, x='Status', y='Copper', hue='Status');
sns.boxplot(ax=axes[2], data=train, x='Status', y='Alk_Phos', hue='Status');

plt.show()
