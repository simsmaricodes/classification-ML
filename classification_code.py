#!/usr/bin/env python
# coding: utf-8

# <h1> Classification Model Development </h1>
# <h2> Simbarashe Mariwande </h2>
# <h3> MSBA 2 </h3>
# <br>In an effort to diversify their revenue stream, Apprentice Chef, Inc. has launched Halfway There,
# a cross-selling promotion where subscribers receive a half bottle of wine from a local California
# vineyard every Wednesday (halfway through the work week). The executives at Apprentice Chef
# also believe this endeavor will create a competitive advantage based on its unique product
# offering of hard to find local wines from smaller vineyards.
# <br>
#
# Halfway There has been exclusively offered to all of the customers in the dataset you received,
# and the executives would like to promote this service to a wider audience. They have tasked you
# with analyzing their data, developing your top insights, and building a machine learning model to
# predict which customers will subscribe to this service.
# <br>

# <h1> Data and Library Imports

# In[1]:


# importing libraries
import random as rand  # random number gen
import numpy as np # data tools
import pandas as pd  # data science essentials
import matplotlib.pyplot as plt  # data visualization
import seaborn as sns  # enhanced data viz
from sklearn.model_selection import train_test_split  # train-test split
from sklearn.linear_model import LogisticRegression  # logistic regression
import statsmodels.formula.api as smf  # logistic regression
from sklearn.metrics import confusion_matrix  # confusion matrix
from sklearn.metrics import roc_auc_score  # auc score
from sklearn.neighbors import KNeighborsClassifier  # KNN for classification
from sklearn.neighbors import KNeighborsRegressor  # KNN for regression
from sklearn.preprocessing import StandardScaler  # standard scaler

# libraries for classification trees
from sklearn.tree import DecisionTreeClassifier  # classification trees
from sklearn.tree import export_graphviz  # exports graphics
from six import StringIO  # saves objects in memory
from IPython.display import Image  # displays on frontend
import pydotplus  # interprets dot objects

from sklearn.model_selection import RandomizedSearchCV  # hyperparameter tuning
from sklearn.metrics import make_scorer  # customizable scorer
from sklearn.metrics import confusion_matrix  # confusion matrix

# new packages
from sklearn.ensemble import RandomForestClassifier  # random forest
from sklearn.ensemble import GradientBoostingClassifier  # gbm

# setting pandas print options
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Filepath
file = './Apprentice_Chef_Dataset.xlsx'

# Importing the dataset
apprentice = pd.read_excel(io=file)


# <h1> Function Defintions

# In[2]:


def text_split_feature(col, df, sep=' ', new_col_name='number_of_names'):
    """
Splits values in a string Series (as part of a DataFrame) and sums the number
of resulting items. Automatically appends summed column to original DataFrame.

PARAMETERS
----------
col          : column to split
df           : DataFrame where column is located
sep          : string sequence to split by, default ' '
new_col_name : name of new column after summing split, default
               'number_of_names'
"""

    df[new_col_name] = 0

    for index, val in df.iterrows():
        df.loc[index, new_col_name] = len(df.loc[index, col].split(sep=' '))


# Splitting the names and summing the number of resulting items
text_split_feature(col='NAME',
                   df=apprentice)


# In[3]:


########################################
# optimal_neighbors
########################################
def optimal_neighbors(X_data,
                      y_data,
                      standardize=True,
                      pct_test=0.25,
                      seed=219,
                      response_type='reg',
                      max_neighbors=30,
                      show_viz=False):
    """
Exhaustively compute training and testing results for KNN across
[1, max_neighbors]. Outputs the maximum test score and (by default) a
visualization of the results.
PARAMETERS
----------
X_data        : explanatory variable data
y_data        : response variable
standardize   : whether or not to standardize the X data, default True
pct_test      : test size for training and validation from (0,1), default 0.25
seed          : random seed to be used in algorithm, default 219
response_type : type of neighbors algorithm to use, default 'reg'
    Use 'reg' for regression (KNeighborsRegressor)
    Use 'class' for classification (KNeighborsClassifier)
max_neighbors : maximum number of neighbors in exhaustive search, default 20
show_viz      : display or surpress k-neigbors visualization, default True
"""

    if standardize == True:
        # optionally standardizing X_data
        scaler = StandardScaler()
        scaler.fit(X_data)
        X_scaled = scaler.transform(X_data)
        X_scaled_df = pd.DataFrame(X_scaled)
        X_data = X_scaled_df

    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                        y_data,
                                                        test_size=pct_test,
                                                        random_state=seed)

    # creating lists for training set accuracy and test set accuracy
    training_accuracy = []
    test_accuracy = []

    # setting neighbor range
    neighbors_settings = range(1, max_neighbors + 1)

    for n_neighbors in neighbors_settings:
        # building the model based on response variable type
        if response_type == 'reg':
            clf = KNeighborsRegressor(n_neighbors=n_neighbors)
            clf.fit(X_train, y_train)

        elif response_type == 'class':
            clf = KNeighborsClassifier(n_neighbors=n_neighbors)
            clf.fit(X_train, y_train)

        else:
            print("Error: response_type must be 'reg' or 'class'")

        # recording the training set accuracy
        training_accuracy.append(clf.score(X_train, y_train))

        # recording the generalization accuracy
        test_accuracy.append(clf.score(X_test, y_test))

    # optionally displaying visualization
    if show_viz == True:
        # plotting the visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.plot(neighbors_settings, training_accuracy, label="training accuracy")
        plt.plot(neighbors_settings, test_accuracy, label="test accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("n_neighbors")
        plt.legend()
        plt.show()

    # returning optimal number of neighbors
    print(f"The optimal number of neighbors is: {test_accuracy.index(max(test_accuracy)) + 1}")
    return test_accuracy.index(max(test_accuracy)) + 1


########################################
# visual_cm
########################################
def visual_cm(true_y, pred_y, labels=None):
    """
Creates a visualization of a confusion matrix.

PARAMETERS
----------
true_y : true values for the response variable
pred_y : predicted values for the response variable
labels : , default None
    """
    # visualizing the confusion matrix

    # setting labels
    lbls = labels

    # declaring a confusion matrix object
    cm = confusion_matrix(y_true=true_y,
                          y_pred=pred_y)

    # heatmap
    sns.heatmap(cm,
                annot=True,
                xticklabels=lbls,
                yticklabels=lbls,
                cmap='Blues',
                fmt='g')

    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix of the Classifier')
    plt.show()


# In[4]:


########################################
# display_tree
########################################
def display_tree(tree, feature_df, height=500, width=800):
    """
    PARAMETERS
    ----------
    tree       : fitted tree model object
        fitted CART model to visualized
    feature_df : DataFrame
        DataFrame of explanatory features (used to generate labels)
    height     : int, default 500
        height in pixels to which to constrain image in html
    width      : int, default 800
        width in pixels to which to constrain image in html
    """

    # visualizing the tree
    dot_data = StringIO()

    # exporting tree to graphviz
    export_graphviz(decision_tree=tree,
                    out_file=dot_data,
                    filled=True,
                    rounded=True,
                    special_characters=True,
                    feature_names=feature_df.columns)

    # declaring a graph object
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

    # creating image
    img = Image(graph.create_png(),
                height=height,
                width=width)

    return img


########################################
# plot_feature_importances
########################################
def plot_feature_importances(model, train, export=False):
    """
    Plots the importance of features from a CART model.

    PARAMETERS
    ----------
    model  : CART model
    train  : explanatory variable training data
    export : whether or not to export as a .png image, default False
    """

    # declaring the number
    n_features = x_train.shape[1]

    # setting plot window
    fig, ax = plt.subplots(figsize=(12, 9))

    plt.barh(range(n_features), model.feature_importances_, align='center')
    plt.yticks(pd.np.arange(n_features), train.columns)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")

    if export == True:
        plt.savefig('Tree_Leaf_50_Feature_Importance.png')


# <h1> Feature Engineering

# In[5]:


# Game of throme names
got_name = ['Sand', 'Stark', 'Martell', 'Greyjoy',
            'Tully', 'Snow', 'Lannister', 'Baratheon',
            'Frey', 'Tyrell', 'Targaryen', 'Arryn']

apprentice['GOT'] = 0

# LOop
for index, value in apprentice.iterrows():

    # Placing in the new list
    if apprentice.loc[index, 'FAMILY_NAME'] in got_name:
        apprentice.loc[index, 'GOT'] = 1

# log transforming Sale_Price and saving it to the dataset
apprentice['log_REVENUE'] = np.log10(apprentice['REVENUE'])

# Log transforms

inter_list = ['LARGEST_ORDER_SIZE', 'PRODUCT_CATEGORIES_VIEWED', 'PC_LOGINS',
              'TOTAL_MEALS_ORDERED', 'UNIQUE_MEALS_PURCH', 'CONTACTS_W_CUSTOMER_SERVICE']

for item in inter_list:
    # Converting to logs and seeing if the data improves
    apprentice['log_' + item] = np.log10(apprentice[item])

# Converting to logs and seeing if the data improves
apprentice['log_clicks'] = np.log10(apprentice['AVG_CLICKS_PER_VISIT'])  # Average clicks log
apprentice['log_customer'] = np.log10(apprentice['CONTACTS_W_CUSTOMER_SERVICE'])  # Customer contact

# In[6]:


# Creating columns for totals

apprentice['Total_Cancellations'] = apprentice['CANCELLATIONS_BEFORE_NOON'] + apprentice['CANCELLATIONS_AFTER_NOON']

apprentice['Total_Logins'] = apprentice['PC_LOGINS'] + apprentice['MOBILE_LOGINS']

# Creating flags for the new variables
apprentice['out_Total_Cancellations'] = 0
apprentice['out_Total_Logins'] = 0

# Summing the totals
for index, value in apprentice.iterrows():

    # Out cancellations
    if apprentice.loc[index, 'Total_Cancellations'] == 0:
        apprentice.loc[index, 'out_Total_Cancellations'] = 1

        # Out Logins
    if apprentice.loc[index, 'Total_Logins'] >= 7:
        apprentice.loc[index, 'out_Total_Logins'] = 1

    # Average order size in dollars
apprentice['AVG_ORDER_REV'] = apprentice['REVENUE'] / apprentice['TOTAL_MEALS_ORDERED']

# Average money spent on a meal
apprentice['AVG_ORDER_SIZE_REV'] = apprentice['REVENUE'] / apprentice['UNIQUE_MEALS_PURCH']

# Gini suggestions
apprentice['Average_Money_Spent'] = 0
apprentice['Average_Spent_per_meal'] = 0

for index, value in apprentice.iterrows():

    if apprentice.loc[index, 'AVG_ORDER_REV'] <= 110:
        apprentice.loc[index, 'Average_Money_Spent'] = 1

    if apprentice.loc[index, 'AVG_ORDER_SIZE_REV'] >= 4000:
        apprentice.loc[index, 'Average_Spent_per_meal'] = 1

    # In[7]:

# STEP 1: splitting personal emails

# placeholder list
placeholder_lst = []

# looping over each email address
for index, col in apprentice.iterrows():
    # splitting email domain at '@'
    split_email = apprentice.loc[index, 'EMAIL'].split(sep='@')

    # appending placeholder_lst with the results
    placeholder_lst.append(split_email)

# converting placeholder_lst into a DataFrame
email_df = pd.DataFrame(placeholder_lst)

# STEP 2: concatenating with original DataFrame
# renaming column to concatenate
email_df.columns = ['0', 'personal_email_domain']

# concatenating personal_email_domain with friends DataFrame
apprentice = pd.concat([apprentice, email_df['personal_email_domain']],
                       axis=1)

# email domain types
personal_email_domains = ['@gmail.com', '@yahoo.com', '@protonmail.com']

# Other Emails
other_email_domains = ['@me.com', '@aol.com', '@live.com', '@passport.com',
                       '@msn.com', '@hotmail.com']

# Domain list
domain_lst = []

# looping to group observations by domain type
for domain in apprentice['personal_email_domain']:
    if '@' + domain in personal_email_domains:
        domain_lst.append('personal')

    elif '@' + domain in other_email_domains:
        domain_lst.append('other')

    else:
        domain_lst.append('work')

# concatenating with original DataFrame
apprentice['domain_group'] = pd.Series(domain_lst)

# checking results
apprentice['domain_group'].value_counts()

# one hot encoding categorical variables
one_hot_domain = pd.get_dummies(apprentice['domain_group'])

# joining codings together
apprentice = apprentice.join([one_hot_domain])

# Dropping Object Data
apprentice = apprentice.drop(['NAME', 'EMAIL', 'FIRST_NAME',
                              'FAMILY_NAME', 'personal_email_domain',
                              'domain_group'], axis=1)

# In[8]:


# Flags based on Decision tree
# Gini suggestions
apprentice['Before_noon_cancellations'] = 0
apprentice['No_personal_usage'] = 0
apprentice['Low_customer_service_cont'] = 0
apprentice['Low_AVG_click'] = 0
apprentice['Total_meals'] = 0
apprentice['Late_delivery'] = 0
apprentice['Low_revenue'] = 0

# For Loop based on Decision Tree
for index, value in apprentice.iterrows():

    # Noon cancellations <1.5
    if apprentice.loc[index, 'CANCELLATIONS_BEFORE_NOON'] <= 1.5:
        apprentice.loc[index, 'Before_noon_cancellations'] = 1

    # Work below 1.5
    if apprentice.loc[index, 'work'] <= 0.5:
        apprentice.loc[index, 'No_personal_usage'] = 1

    # WCustomer service low
    if apprentice.loc[index, 'log_customer'] <= 0.812:
        apprentice.loc[index, 'Low_customer_service_cont'] = 1

    # Clicks low
    if apprentice.loc[index, 'log_clicks'] <= 1.267:
        apprentice.loc[index, 'Low_AVG_click'] = 1

    # low total meals
    if apprentice.loc[index, 'TOTAL_MEALS_ORDERED'] <= 47.5:
        apprentice.loc[index, 'Total_meals'] = 1

    # late delivery
    if apprentice.loc[index, 'LATE_DELIVERIES'] <= 1.5:
        apprentice.loc[index, 'Late_delivery'] = 1

    # Low revenue
    if apprentice.loc[index, 'REVENUE'] <= 3987:
        apprentice.loc[index, 'Low_revenue'] = 1

# In[9]:


# Dummy Variables for the factors we found above with at leasst 100 observations
apprentice['Visit_time'] = 0
apprentice['after_canc'] = 0
apprentice['weekly_plan_sub'] = 0
apprentice['early_delivery'] = 0
apprentice['late_delivery'] = 0
apprentice['masterclass_att'] = 0
apprentice['view_photo'] = 0
apprentice['noon_canc'] = 0

# Iter over eachg column to get the new boolean feature columns
for index, value in apprentice.iterrows():

    # For noon cancellations
    if apprentice.loc[index, 'AVG_TIME_PER_SITE_VISIT'] > apprentice['AVG_TIME_PER_SITE_VISIT'].median():
        apprentice.loc[index, 'Visit_time'] = 1

    # For afternoon cancelations
    if apprentice.loc[index, 'CANCELLATIONS_AFTER_NOON'] > 0:
        apprentice.loc[index, 'after_canc'] = 1

    # Weekly meal plan subscription
    if apprentice.loc[index, 'WEEKLY_PLAN'] > 0:
        apprentice.loc[index, 'weekly_plan_sub'] = 1

    # Early deliveries
    if apprentice.loc[index, 'EARLY_DELIVERIES'] > 0:
        apprentice.loc[index, 'early_delivery'] = 1

    # Late Deliveries
    if apprentice.loc[index, 'LATE_DELIVERIES'] > 0:
        apprentice.loc[index, 'late_delivery'] = 1

    # Masterclass attendance
    if apprentice.loc[index, 'MASTER_CLASSES_ATTENDED'] > 0:
        apprentice.loc[index, 'masterclass_att'] = 1

    # Viewed Photos
    if apprentice.loc[index, 'TOTAL_PHOTOS_VIEWED'] > 0:
        apprentice.loc[index, 'view_photo'] = 1

    if apprentice.loc[index, 'CANCELLATIONS_BEFORE_NOON'] > 0:
        apprentice.loc[index, 'noon_canc'] = 1

# Iter over eachg column to get the new boolean feature columns

for index, value in apprentice.iterrows():

    # Total meals greater than 200
    if apprentice.loc[index, 'TOTAL_MEALS_ORDERED'] >= 200:
        apprentice.loc[index, 'meals_below_fif'] = 1

    # Total meals less than 15
    if apprentice.loc[index, 'TOTAL_MEALS_ORDERED'] <= 15:
        apprentice.loc[index, 'meals_above_two'] = 1

    # Unique meals greater 10
    if apprentice.loc[index, 'UNIQUE_MEALS_PURCH'] > 10:
        apprentice.loc[index, 'unique_meals_above_ten'] = 1

    # Customer service less than 10
    if apprentice.loc[index, 'CONTACTS_W_CUSTOMER_SERVICE'] < 10:
        apprentice.loc[index, 'cust_serv_under_ten'] = 1

    # Clicks below 8
    if apprentice.loc[index, 'AVG_CLICKS_PER_VISIT'] < 8:
        apprentice.loc[index, 'click_under_eight'] = 1

# Adding the new variable
apprentice['freq_customer_service'] = 0

# Instantiating dummy variables
for index, value in apprentice.iterrows():

    # For noon cancellations
    if apprentice.loc[index, 'CONTACTS_W_CUSTOMER_SERVICE'] > (apprentice.loc[index, 'TOTAL_MEALS_ORDERED']) / 2:
        apprentice.loc[index, 'freq_customer_service'] = 1

# Checking distribution
contact_greater = []
mobile_greater = []

# Instantiating dummy variables
for index, value in apprentice.iterrows():

    # For noon cancellations
    if apprentice.loc[index, 'CONTACTS_W_CUSTOMER_SERVICE'] > (apprentice.loc[index, 'TOTAL_MEALS_ORDERED']) / 2:
        contact_greater.append(1)
    else:
        contact_greater.append(0)

# Instantiating dummy variables
for index, value in apprentice.iterrows():

    if apprentice.loc[index, 'MOBILE_LOGINS'] > apprentice.loc[index, 'PC_LOGINS']:
        mobile_greater.append(1)

    else:
        mobile_greater.append(0)

contact_greater = pd.DataFrame(contact_greater)
mobile_greater = pd.DataFrame(mobile_greater)

contact_greater.value_counts()  # Checking distribution of zeros

# Adding them to the data
apprentice['contact_greater'] = contact_greater
apprentice['mobile_greater'] = mobile_greater

# In[10]:


# Decision TreeeAd ons
apprentice['other_lo'] = 0
apprentice['canc_no_lo'] = 0
apprentice['avg_vist_lo'] = 0
apprentice['mobi_num'] = 0

# Iter over eachg column to get the new boolean feature columns
for index, value in apprentice.iterrows():

    # For noon cancellations
    if apprentice.loc[index, 'AVG_TIME_PER_SITE_VISIT'] <= 14.8:
        apprentice.loc[index, 'avg_vist_lo'] = 1

    # For afternoon cancelations
    if apprentice.loc[index, 'CANCELLATIONS_BEFORE_NOON'] <= 0.5:
        apprentice.loc[index, 'canc_no_lo'] = 1

    # Weekly meal plan subscription
    if apprentice.loc[index, 'other'] <= 0.5:
        apprentice.loc[index, 'other_lo'] = 1

    # Early deliveries
    if apprentice.loc[index, 'MOBILE_NUMBER'] <= 0.5:
        apprentice.loc[index, 'mobi_num'] = 1

# In[11]:


# More features
apprentice['other_many_names'] = 0

for index, value in apprentice.iterrows():

    if apprentice.loc[index, 'other'] == 1 and apprentice.loc[index, 'number_of_names'] == 1:
        apprentice.loc[index, 'other_many_names'] = 1

apprentice['other_noon_cancellations'] = 0

for index, value in apprentice.iterrows():

    if apprentice.loc[index, 'other'] == 1 and apprentice.loc[index, 'Before_noon_cancellations'] == 1:
        apprentice.loc[index, 'other_noon_cancellations'] = 1

apprentice['other_mobs'] = 0

for index, value in apprentice.iterrows():

    if apprentice.loc[index, 'other'] == 1 and apprentice.loc[index, 'mobi_num'] == 1:
        apprentice.loc[index, 'other_mobs'] = 1

apprentice['mob_late'] = 0

for index, value in apprentice.iterrows():

    if apprentice.loc[index, 'late_delivery'] == 1 and apprentice.loc[index, 'mobi_num'] == 1:
        apprentice.loc[index, 'mob_late'] = 1

# <h1> Model Building

# In[12]:


# train/test split with the full model
apprentice_data = apprentice.drop('CROSS_SELL_SUCCESS', axis=1)
apprentice_target = apprentice.loc[:, 'CROSS_SELL_SUCCESS']

# This is the exact code we were using before
x_train, x_test, y_train, y_test = train_test_split(
    apprentice_data,
    apprentice_target,
    test_size=0.25,
    random_state=219,
    stratify=apprentice_target)

# merging training data for statsmodels
apprentice_train = pd.concat([x_train, y_train], axis=1)

# <h2> Candidate Dictionary

# In[13]:


# All the x variable after removing ridiculous values
a_var = ['MOBILE_NUMBER', 'CANCELLATIONS_BEFORE_NOON', 'TASTES_AND_PREFERENCES',
         'EARLY_DELIVERIES', 'REFRIGERATED_LOCKER', 'MASTER_CLASSES_ATTENDED',
         'log_PC_LOGINS', 'number_of_names', 'Total_Cancellations', 'other',
         'Before_noon_cancellations', 'No_personal_usage', 'Low_customer_service_cont',
         'Low_AVG_click', 'Total_meals', 'GOT']

# Variables were added after looking at the classification tree
b_var = ['TOTAL_MEALS_ORDERED', 'TASTES_AND_PREFERENCES', 'EARLY_DELIVERIES',
         'REFRIGERATED_LOCKER', 'AVG_PREP_VID_TIME', 'GOT', 'log_REVENUE',
         'log_LARGEST_ORDER_SIZE', 'log_PRODUCT_CATEGORIES_VIEWED', 'log_PC_LOGINS',
         'number_of_names', 'Total_Cancellations', 'Total_Logins', 'out_Total_Cancellations',
         'out_Total_Logins', 'Average_Spent_per_meal', 'other', 'Before_noon_cancellations',
         'No_personal_usage', 'Low_AVG_click', 'Total_meals', 'Late_delivery', 'after_canc',
         'weekly_plan_sub', 'early_delivery', 'late_delivery', 'masterclass_att',
         'view_photo', 'freq_customer_service', 'canc_no_lo', 'avg_vist_lo', 'mobi_num']

# Removal to include important variables after feature importance
c_var = ['TOTAL_MEALS_ORDERED', 'CANCELLATIONS_BEFORE_NOON', 'AVG_PREP_VID_TIME', 'GOT',
         'No_personal_usage', 'log_REVENUE', 'number_of_names', 'other', 'Before_noon_cancellations',
         'No_personal_usage', 'other_many_names', 'Total_meals', 'masterclass_att']

# Different variable combinbations inspired by new features created
d_var = ['TASTES_AND_PREFERENCES', 'EARLY_DELIVERIES', 'REFRIGERATED_LOCKER',
         'AVG_PREP_VID_TIME', 'log_REVENUE', 'log_PC_LOGINS', 'number_of_names',
         'Total_Logins', 'Average_Spent_per_meal', 'other', 'Before_noon_cancellations',
         'No_personal_usage', 'Low_AVG_click', 'Total_meals', 'Late_delivery', 'after_canc',
         'early_delivery', 'late_delivery', 'masterclass_att', 'freq_customer_service', 'canc_no_lo',
         'avg_vist_lo', 'mobi_num', 'mob_late']

e_var = ['TASTES_AND_PREFERENCES', 'EARLY_DELIVERIES', 'REFRIGERATED_LOCKER', 'log_PC_LOGINS',
         'number_of_names', 'GOT', 'Total_Logins', 'other', 'Before_noon_cancellations',
         'No_personal_usage', 'Low_AVG_click', 'Total_meals', 'early_delivery', 'masterclass_att',
         'freq_customer_service', 'canc_no_lo', 'mobi_num']

# <h1> Models

# <h2> Model Performance Data Frame

# In[14]:


# creating a dictionary for model results
model_performance = {

    'Model Name': [],

    'AUC Score': [],

    'Training Accuracy': [],

    'Testing Accuracy': [],

    'Confusion Matrix': []}

# converting model_performance into a DataFrame
model_performance = pd.DataFrame(model_performance)

# <h2> Logistic Regression Model

# In[15]:


# train/test split with the logit_sig variables
apprentice_data = apprentice.loc[:, c_var]
apprentice_target = apprentice.loc[:, 'CROSS_SELL_SUCCESS']

# train/test split
x_train, x_test, y_train, y_test = train_test_split(
    apprentice_data,
    apprentice_target,
    random_state=219,
    test_size=0.25,
    stratify=apprentice_target)

# building a model based on hyperparameter tuning results

# INSTANTIATING a logistic regression model with tuned values
lr_tuned = LogisticRegression(solver='lbfgs',
                              random_state=219,
                              C=3.1,
                              warm_start=False,
                              max_iter=1000)

# FIT
lr_tuned = lr_tuned.fit(x_train, y_train)

# PREDICTING based on the testing set
lr_tuned_pred = lr_tuned.predict(x_test)

# SCORING the results
print('LR Tuned Training ACCURACY:', lr_tuned.score(x_train, y_train).round(4))
print('LR Tuned Testing  ACCURACY:', lr_tuned.score(x_test, y_test).round(4))
print('LR Tuned AUC Score        :', roc_auc_score(y_true=y_test,
                                                   y_score=lr_tuned_pred).round(4))

# saving scoring data for future use
lr_tuned_train_score = lr_tuned.score(x_train, y_train).round(4)  # accuracy
lr_tuned_test_score = lr_tuned.score(x_test, y_test).round(4)  # accuracy

# saving the AUC score
lr_tuned_auc = roc_auc_score(y_true=y_test,
                             y_score=lr_tuned_pred).round(4)  # auc
# unpacking the confusion matrix
lr_tuned_tn, lr_tuned_fp, lr_tuned_fn, lr_tuned_tp = confusion_matrix(y_true=y_test, y_pred=lr_tuned_pred).ravel()

# printing each result one-by-one
print(f"""
True Negatives : {lr_tuned_tn}
False Positives: {lr_tuned_fp}
False Negatives: {lr_tuned_fn}
True Positives : {lr_tuned_tp}
""")

# declaring model performance objects
lr_train_acc = lr_tuned.score(x_train, y_train).round(4)
lr_test_acc = lr_tuned.score(x_test, y_test).round(4)
lr_auc = roc_auc_score(y_true=y_test,
                       y_score=lr_tuned_pred).round(4)

# appending to model_performance
model_performance = model_performance.append(
    {'Model Name': 'Tuned Logistic Regression',
     'Training Accuracy': lr_train_acc,
     'Testing Accuracy': lr_test_acc,
     'AUC Score': lr_auc,
     'Confusion Matrix': (lr_tuned_tn,
                          lr_tuned_fp,
                          lr_tuned_fn,
                          lr_tuned_tp)},
    ignore_index=True)

# <h2> KNN

# In[16]:


# train/test split with the logit_sig variables
apprentice_data = apprentice.loc[:, c_var]
apprentice_target = apprentice.loc[:, 'CROSS_SELL_SUCCESS']

# train/test split
x_train, x_test, y_train, y_test = train_test_split(
    apprentice_data,
    apprentice_target,
    random_state=219,
    test_size=0.25,
    stratify=apprentice_target)
# determining the optimal number of neighbors
opt_neighbors = optimal_neighbors(X_data=apprentice_data,
                                  y_data=apprentice_target,
                                  response_type='class')

# INSTANTIATING StandardScaler()
scaler = StandardScaler()

# FITTING the data
scaler.fit(apprentice_data)

# TRANSFORMING the data
X_scaled = scaler.transform(apprentice_data)

# converting to a DataFrame
X_scaled_df = pd.DataFrame(X_scaled)

# train-test split with the scaled data
X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    X_scaled_df,
    apprentice_target,
    random_state=219,
    test_size=0.25,
    stratify=apprentice_target)

# INSTANTIATING a KNN classification model with optimal neighbors
knn_opt = KNeighborsClassifier(n_neighbors=opt_neighbors)

# FITTING the training data
knn_fit = knn_opt.fit(X_train_scaled, y_train_scaled)

# PREDICTING based on the testing set
knn_pred = knn_fit.predict(X_test_scaled)

# SCORING the results
print('Training ACCURACY:', knn_fit.score(X_train_scaled, y_train_scaled).round(4))
print('Testing  ACCURACY:', knn_fit.score(X_test_scaled, y_test_scaled).round(4))
print('AUC Score        :', roc_auc_score(y_true=y_test,
                                          y_score=knn_pred).round(4))

# saving scoring data
knn_train_score = knn_fit.score(X_train_scaled, y_train_scaled).round(4)
knn_test_score = knn_fit.score(X_test_scaled, y_test_scaled).round(4)

# saving AUC score
knn_auc_score = roc_auc_score(y_true=y_test,
                              y_score=knn_pred).round(4)
# unpacking the confusion matrix
knn_tree_tn, knn_tree_fp, knn_tree_fn, knn_tree_tp = confusion_matrix(y_true=y_test, y_pred=knn_pred).ravel()

# appending to model_performance
model_performance = model_performance.append(
    {'Model Name': 'KNN Classification',
     'Training Accuracy': knn_train_score,
     'Testing Accuracy': knn_test_score,
     'AUC Score': knn_auc_score,
     'Confusion Matrix': (knn_tree_tn,
                          knn_tree_fp,
                          knn_tree_fn,
                          knn_tree_tp)},
    ignore_index=True)

# <h2>Optimised Tree

# In[20]:


# train/test split with the logit_sig variables
apprentice_data = apprentice.loc[:, c_var]
apprentice_target = apprentice.loc[:, 'CROSS_SELL_SUCCESS']

# train/test split
x_train, x_test, y_train, y_test = train_test_split(
    apprentice_data,
    apprentice_target,
    random_state=219,
    test_size=0.25,
    stratify=apprentice_target)

# INSTANTIATING a classification tree object
pruned_tree = DecisionTreeClassifier(max_depth=3,
                                     min_samples_leaf=5,
                                     random_state=219)

# FITTING the training data
pruned_tree_fit = pruned_tree.fit(apprentice_data, apprentice_target)

# PREDICTING on new data
pruned_tree_pred = pruned_tree_fit.predict(x_test)

# SCORING the model
print('Training ACCURACY:', pruned_tree_fit.score(x_train, y_train).round(4))
print('Testing  ACCURACY:', pruned_tree_fit.score(x_test, y_test).round(4))
print('AUC Score        :', roc_auc_score(y_true=y_test,
                                          y_score=pruned_tree_pred).round(4))

# unpacking the confusion matrix
tuned_tree_tn, tuned_tree_fp, tuned_tree_fn, tuned_tree_tp = confusion_matrix(y_true=y_test,
                                                                              y_pred=pruned_tree_pred).ravel()

# declaring model performance objects
tree_train_acc = pruned_tree.score(x_train, y_train).round(4)
tree_test_acc = pruned_tree.score(x_test, y_test).round(4)
tree_auc = roc_auc_score(y_true=y_test,
                         y_score=pruned_tree_pred).round(4)

# appending to model_performance
model_performance = model_performance.append(
    {'Model Name': 'Tuned Classification Tree',
     'Training Accuracy': tree_train_acc,
     'Testing Accuracy': tree_test_acc,
     'AUC Score': tree_auc,
     'Confusion Matrix': (tuned_tree_tn,
                          tuned_tree_fp,
                          tuned_tree_fn,
                          tuned_tree_tp)},
    ignore_index=True)

# <h2> Random Forrest

# In[19]:


# train/test split with the logit_sig variables
apprentice_data = apprentice.loc[:, a_var]
apprentice_target = apprentice.loc[:, 'CROSS_SELL_SUCCESS']

# train/test split
x_train, x_test, y_train, y_test = train_test_split(
    apprentice_data,
    apprentice_target,
    random_state=219,
    test_size=0.25,
    stratify=apprentice_target)

# to avoid running another RandomizedSearch
forest_tuned = RandomForestClassifier(criterion='entropy',
                                      random_state=219,
                                      warm_start=True,
                                      min_samples_leaf=11,
                                      n_estimators=350,
                                      max_depth=5)

# FITTING the model object
forest_tuned_fit = forest_tuned.fit(apprentice_data, apprentice_target)

# PREDICTING based on the testing set
forest_tuned_pred = forest_tuned_fit.predict(x_test)

# SCORING the results
print('Forest Tuned Training ACCURACY:', forest_tuned.score(x_train, y_train).round(4))
print('Forest Tuned Testing  ACCURACY:', forest_tuned.score(x_test, y_test).round(4))
print('Forest Tuned AUC Score        :', roc_auc_score(y_true=y_test,
                                                       y_score=forest_tuned_pred).round(4))

# saving scoring data for future use
forest_tuned_train_score = forest_tuned.score(x_train, y_train).round(4)  # accuracy
forest_tuned_test_score = forest_tuned.score(x_test, y_test).round(4)  # accuracy

# saving the AUC score
forest_tuned_auc = roc_auc_score(y_true=y_test,
                                 y_score=forest_tuned_pred).round(4)  # auc
# unpacking the confusion matrix
rf_tn, rf_fp, rf_fn, rf_tp = confusion_matrix(y_true=y_test, y_pred=forest_tuned_pred).ravel()

# declaring model performance objects
rf_train_acc = forest_tuned_fit.score(x_train, y_train).round(4)
rf_test_acc = forest_tuned_fit.score(x_test, y_test).round(4)
rf_auc = roc_auc_score(y_true=y_test,
                       y_score=forest_tuned_pred).round(4)

# appending to model_performance
model_performance = model_performance.append(
    {'Model Name': 'Random Forest (Sig)',
     'Training Accuracy': rf_train_acc,
     'Testing Accuracy': rf_test_acc,
     'AUC Score': rf_auc,
     'Confusion Matrix': (rf_tn,
                          rf_fp,
                          rf_fn,
                          rf_tp)},
    ignore_index=True)

# <h2> GBM

# In[19]:


# train/test split with the logit_sig variables
apprentice_data = apprentice.loc[:, b_var]
apprentice_target = apprentice.loc[:, 'CROSS_SELL_SUCCESS']

# train/test split
x_train, x_test, y_train, y_test = train_test_split(
    apprentice_data,
    apprentice_target,
    random_state=219,
    test_size=0.25,
    stratify=apprentice_target)

# INSTANTIATING the model object without hyperparameters
full_gbm_default = GradientBoostingClassifier(loss='deviance',
                                              learning_rate=0.1,
                                              n_estimators=100,
                                              criterion='friedman_mse',
                                              max_depth=2,
                                              warm_start=False,
                                              random_state=219)

# FIT step is needed as we are not using .best_estimator
full_gbm_default_fit = full_gbm_default.fit(x_train, y_train)

# PREDICTING based on the testing set
full_gbm_default_pred = full_gbm_default_fit.predict(x_test)

# SCORING the results
print('Training ACCURACY:', full_gbm_default_fit.score(x_train, y_train).round(4))
print('Testing ACCURACY :', full_gbm_default_fit.score(x_test, y_test).round(4))
print('AUC Score        :', roc_auc_score(y_true=y_test,
                                          y_score=full_gbm_default_pred).round(4))
# unpacking the confusion matrix
gbm_default_tn, gbm_default_fp, gbm_default_fn, gbm_default_tp = confusion_matrix(y_true=y_test,
                                                                                  y_pred=full_gbm_default_pred).ravel()

# SCORING the model
gbm_train_acc = full_gbm_default_fit.score(x_train, y_train).round(4)
gbm_test_acc = full_gbm_default_fit.score(x_test, y_test).round(4)
gbm_auc = roc_auc_score(y_true=y_test,
                        y_score=full_gbm_default_pred).round(4)

# appending to model_performance
model_performance = model_performance.append(
    {'Model Name': 'GBM (Full)',
     'Training Accuracy': gbm_train_acc,
     'Testing Accuracy': gbm_test_acc,
     'AUC Score': gbm_auc,
     'Confusion Matrix': (gbm_default_tn,
                          gbm_default_fp,
                          gbm_default_fn,
                          gbm_default_tp)},
    ignore_index=True)

# <h1> Model Performance

# In[20]:


model_performance.sort_values('AUC Score', ascending=False)

# <h1> Selected Model

# In[21]:


print(model_performance.loc[2, :])


