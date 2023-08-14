#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.ticker as ticker
import matplotlib.style as style
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import plotly.graph_objs as go
import plotly.offline as py
from collections import Counter

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression 
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet

from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error

from scipy import stats
from scipy.stats import skew


# In[2]:


# Read the dataset

data = pd.read_csv('C:/Project/rollingsales.csv', skipinitialspace=True)
data.head()


# In[3]:


# Describe the dataset

print (f"The dataset has {data.shape[0]} rows and {data.shape[1]} columns")
print ('-'*100)
data.info()


# In[4]:


# Delete unwanted columns

del data['EASEMENT'] # empty column
del data['APARTMENT NUMBER'] # empty column
del data['LOT'] # not required

#data.head()


# In[5]:


# Specifying numeric variables

numeric = ["RESIDENTIAL UNITS","COMMERCIAL UNITS","TOTAL UNITS", "LAND SQUARE FEET" ,
           "GROSS SQUARE FEET", "SALE PRICE"]
for col in numeric: 
    data[col] = pd.to_numeric(data[col], errors='coerce')
    
data.describe()


# In[6]:


# Mapping borough names to key value

data['BOROUGH'] = data['BOROUGH'].map({1:'Manhattan', 2:'Bronx', 3: 'Brooklyn', 4:'Queens',5:'Staten Island'})


# In[7]:


# Formatting date column and extracting the month and year

data['SALE DATE'] = pd.to_datetime(data['SALE DATE'], errors='coerce') 

data['SALE YEAR'] = pd.DatetimeIndex(data['SALE DATE']).year
data['SALE MONTH'] = pd.DatetimeIndex(data['SALE DATE']).month


# In[8]:


del data['SALE DATE'] # not required any more


# In[9]:


# Calculating building age at the time of sale

data['BUILDING AGE'] = data['SALE YEAR'] - data['YEAR BUILT']
data.loc[data['YEAR BUILT'] == 0, 'BUILDING AGE'] = pd.NA


# In[10]:


data_raw = data.copy() # creating a copy


# In[11]:


# Checking missing values

data.replace(' ',np.nan, inplace=True)
round(data.isna().sum() /len(data) *100,2)


# #### Fill in missing values of 'Land sqft' and 'Gross sqft' using Imputation method

# In[80]:


# Check Land sqft anf Gross sqft columns if one of them is null

land_count = ((data['LAND SQUARE FEET'].isnull()) & (data['GROSS SQUARE FEET'].notnull())).sum()
gross_count = ((data['LAND SQUARE FEET'].notnull()) & (data['GROSS SQUARE FEET'].isnull())).sum()

print(land_count)
print(gross_count)
print(land_count + gross_count, 'rows can be filled by imputation method.')


# In[14]:


# Create a new dataframe to perform imputation on the missing Land sqft data

imp = data[['LAND SQUARE FEET','GROSS SQUARE FEET']]
imp.dropna(inplace = True)

imp.head()


# In[15]:


# Fit a regression model to predict the Land sqft

x = imp['GROSS SQUARE FEET']
y = imp['LAND SQUARE FEET']

model = LinearRegression()
reg = model.fit(x.values.reshape(-1,1),y.values.reshape(-1,1))

print("Regression Model: y = {}*x + {}".format(reg.coef_, reg.intercept_))


# In[16]:


# Define a function to calculate Land sqft 

def imp_land(x):
    return x * 0.63778678 + 1033.76397579


# In[17]:


# Fit a regression model to predict the Gross sqft

x = imp['LAND SQUARE FEET']
y = imp['GROSS SQUARE FEET']

model = LinearRegression()
reg = model.fit(x.values.reshape(-1,1),y.values.reshape(-1,1))

print("Regression Model: y = {}*x + {}".format(reg.coef_, reg.intercept_))


# In[18]:


# Define a function to calculate Gross sqft 

def imp_gross(x):
    return x * 0.64815423 + 1701.2484797


# In[19]:


# Impute missing values of these columns.

data['LAND SQUARE FEET'] = data['LAND SQUARE FEET'].mask((data['LAND SQUARE FEET'].isnull()) & 
                                                         (data['GROSS SQUARE FEET'].notnull()), 
                                                         imp_land(data['GROSS SQUARE FEET']))

data['GROSS SQUARE FEET'] = data['GROSS SQUARE FEET'].mask((data['LAND SQUARE FEET'].notnull()) & 
                                                           (data['GROSS SQUARE FEET'].isnull()),
                                                           imp_gross(data['LAND SQUARE FEET']))


# In[20]:


data = data[data["SALE PRICE"] > 0]  # delete zero sale price transactions 
data = data[data["SALE PRICE"].notnull()] # delete NA values

#data.head()


# In[21]:


# Check if there are duplicate values

sum(data.duplicated(data.columns))


# In[22]:


# Delete duplicates

data = data.drop_duplicates(data.columns, keep='last')
sum(data.duplicated(data.columns)) # check for duplicates again


# In[23]:


# Check which columns still have missing data

data.columns[data.isnull().any()]


# In[24]:


# Percent missing values per column

missing = data.isnull().sum()/len(data)
missing = missing[missing > 0]
missing.sort_values(inplace=True)
missing


# In[25]:


# Table of missing percentage

missing = missing.to_frame()
missing.columns = ['count']
missing.index.names = ['Name']
missing['Name'] = missing.index
missing


# In[26]:


# Plot the missing values

sns.set(style='whitegrid', color_codes=True)
sns.barplot(x='Name', y='count', data=missing)
plt.xticks(rotation = 90)
sns


# In[27]:


# Remove data where sum of commercial and residential units don't equal total units

data = data[data['TOTAL UNITS'] == data['COMMERCIAL UNITS'] + data['RESIDENTIAL UNITS']]


# In[28]:


# Sort the total units to check zero total units and outliers

data[["TOTAL UNITS", "SALE PRICE"]].groupby(['TOTAL UNITS'], as_index=False).count().sort_values(by='SALE PRICE', ascending=False)


# In[29]:


# Remove rows with zero total units and one outlier with 2261 units

data = data[(data['TOTAL UNITS'] > 0) & (data['TOTAL UNITS'] != 2261)]


# In[30]:


# Delete rows where year built is zero

data = data[data['YEAR BUILT'] != 0]
#del df['YEAR BUILT']


# In[31]:


# Delete rows where sqft is still zero

data = data[data["LAND SQUARE FEET"] != 0]
data = data[data["GROSS SQUARE FEET"] != 0]


# In[32]:


data.shape


# In[33]:


data_copy = data.copy() # creating a copy


# #### Removing outliers

# In[35]:


round(data.describe([0.75,0.85,0.95,0.99,0.995,0.999]),3)


# In[36]:


data["GROSS SQUARE FEET"].mean() + 2*data["GROSS SQUARE FEET"].std()


# In[37]:


data = data[data["GROSS SQUARE FEET"] < 72000] # deleting outlier that are 2 std from mean 


# In[38]:


data["LAND SQUARE FEET"].mean() + 3*data["LAND SQUARE FEET"].std()


# In[39]:


data = data[data["LAND SQUARE FEET"] < 43000] # deleting outlier that are 3 std from mean  


# In[40]:


# Check if correlation between data increased after handling missing/duplicate values and outlier removal

corr = pd.DataFrame(data.corr().abs().unstack()
                    .sort_values(ascending = False)["SALE PRICE"][1:]).rename(columns = {0:"Corr After Outlier and Missing"})
a = pd.DataFrame(data_copy.corr().abs().unstack()
                 .sort_values(ascending = False)["SALE PRICE"][1:]).rename(columns = {0:"Corr After Missing"})
b = pd.DataFrame(data_raw.corr().abs().unstack()
                 .sort_values(ascending = False)["SALE PRICE"][1:]).rename(columns = {0:"Corr Default"}).iloc[:11,:]

pd.concat([b,a,corr],axis=1).dropna()


# In[41]:


# Check to see if our main variable, Sale Price, is normal

sns.set_style("whitegrid")
plt.figure(figsize=(12,5))
plotd = sns.distplot(data[(data['SALE PRICE']>100) & (data['SALE PRICE'] < 5000000)]['SALE PRICE'], kde=True, bins=100)

tick_spacing=250000
plotd.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
plotd.set_xlim([-100000, 5000000])
plt.xticks(rotation=30)
plt.axvline(data[(data['SALE PRICE']>100) & (data['SALE PRICE'] < 5000000)]['SALE PRICE'].mean(), c='red')
plt.axvline(data[(data['SALE PRICE']>100) & (data['SALE PRICE'] < 5000000)]['SALE PRICE'].median(), c='blue')
plt.text(250000,0.0000012, "median")
plt.text(850000,0.0000010, "mean")
plt.show()


# In[42]:


# Calculate skewness and kurtosis

df = data[(data['SALE PRICE'] > 100) & (data['SALE PRICE'] < 5000000)]

print("Skewness: " + str(df['SALE PRICE'].skew()))
print("Kurtosis: " + str(df['SALE PRICE'].kurt()))


# In[43]:


# Transform the target variable

sales = np.log(data[(data['SALE PRICE'] > 100000) & (data['SALE PRICE'] < 5000000)]['SALE PRICE'])
print(sales.skew())
sns.distplot(sales, bins=30)


# #### Visualizing Numerical Data

# In[45]:


# Boxplot of sale price by boroughs

b1 = go.Box(
    y=df[df.BOROUGH == 'Manhattan']['SALE PRICE'],
    name = 'Manhattan',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
b2 = go.Box(
    y=df[df.BOROUGH == 'Bronx']['SALE PRICE'],
    name = 'Bronx',
    marker = dict(
        color = 'rgb(8,81,156)',
    )
)
b3 = go.Box(
    y=df[df.BOROUGH == 'Brooklyn']['SALE PRICE'],
    name = 'Brooklyn',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)
b4 = go.Box(
    y=df[df.BOROUGH == 'Queens']['SALE PRICE'],
    name = 'Queens',
    marker = dict(
        color = 'rgb(12, 128, 128)',
    )
)
b5 = go.Box(
    y=df[df.BOROUGH == 'Staten Island']['SALE PRICE'],
    name = 'Staten Island',
    marker = dict(
        color = 'rgb(12, 12, 140)',
    )
)

dat = [b1, b2, b3, b4, b5]
layout = go.Layout(
    title='Housing Prices by Boroughs',
    xaxis=dict(
        title='Borough'
    ),
    yaxis=dict(
        title='Sale Price'
    ),
    paper_bgcolor='rgb(243, 243, 243)',
    plot_bgcolor='rgb(243, 243, 243)'
)

fig = go.Figure(data=dat, layout=layout)
py.iplot(fig)


# In[46]:


# Check seasonality in sales trend

season_colors = sns.color_palette("muted", 12)
monthly_mean_sales = df.groupby('SALE MONTH')['SALE PRICE'].mean()

plt.figure(figsize=(10, 6))
monthly_mean_sales.plot(kind='bar', color=season_colors)
plt.xlabel('Month')
plt.ylabel('Mean Sale Price')
plt.title('Mean Sale Price by Month')
plt.xticks(np.arange(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.show()


# In[47]:


# Sale price by year

plt.figure(figsize=(10, 6))
sns.barplot(x=df['SALE YEAR'], y=df['SALE PRICE'])
plt.title("Sale Price by Year")
plt.xlabel("Year")
plt.ylabel("Sale Price")
plt.xticks(rotation=45)
plt.show()


# In[48]:


# Average land sqft and sale price by boroughs

borough_avg = df.groupby('BOROUGH').agg({'LAND SQUARE FEET': 'mean', 'SALE PRICE': 'mean'}).reset_index()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
sns.barplot(x='BOROUGH', y='LAND SQUARE FEET', data=borough_avg, color='blue', ax=ax1)
ax1.set_title("Average Land Square Feet by Borough")
ax1.set_xlabel("Boroughs")
ax1.set_ylabel("Average Land Square Feet")

sns.barplot(x='BOROUGH', y='SALE PRICE', data=borough_avg, color='orange', ax=ax2)
ax2.set_title("Average Sale Price by Borough")
ax2.set_xlabel("Boroughs")
ax2.set_ylabel("Average Sale Price")

plt.tight_layout()
plt.show()


# In[49]:


# Get the list of top 10 neighborhoods by frequency
neighborhoods = list(dict(Counter(df.NEIGHBORHOOD).most_common(10)).keys())

# Calculate average sale prices for each neighborhood
avg_sale_prices = []
for n in neighborhoods:
    avg_price = np.mean(df[df.NEIGHBORHOOD == n]['SALE PRICE'])
    avg_sale_prices.append(avg_price)

# Create the bar plot data
data = [go.Bar(
            y=neighborhoods,
            x=avg_sale_prices,
            width=0.7,
            opacity=0.8, 
            orientation='h',
            marker=dict(
                color='rgb(255, 166, 77)',
                line=dict(
                    color='rgb(255, 97, 0)',
                    width=1.5,
                )
            ),
        )]

layout = go.Layout(
    title='House Price by Top 10 Neighborhoods',
    autosize=True,
    margin=go.Margin(
        l=150,
        r=50,
        b=100,
        t=100,
        pad=4
    ),
    xaxis=dict(
        title='Avg. Sale Price',
        tickprefix="$",
    ),
    yaxis=dict(
        title='Neighborhood',
        tickfont=dict(size=12),
    ),
    paper_bgcolor='rgb(240, 240, 240)',
    plot_bgcolor='rgb(240, 240, 240)',  
)

fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='barplottype')


# In[50]:


# Create correlation heatmap of features

sns.set_style('whitegrid')
plt.subplots(figsize=(20, 15))

mask = np.zeros_like(df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

sns.heatmap(df.corr(),
            cmap=sns.diverging_palette(20, 220, n=200),
            mask=mask,
            annot=True,
            center=0,
            fmt=".2f",
            linewidths=0.5 
           )

plt.title("Correlation Heatmap of Features", fontsize=30)
plt.show()


# #### Visualizing categorical data

# In[52]:


cat_data=df.select_dtypes(exclude=[np.number])
cat_data.describe()


# In[53]:


# Start with TAX CLASS AT PRESENT
df['TAX CLASS AT PRESENT'].unique()


# In[54]:


pivot = df[df['TAX CLASS AT PRESENT'] != ' '].pivot_table(index='TAX CLASS AT PRESENT', values='SALE PRICE', aggfunc=np.median)
pivot


# In[55]:


# Get median sale price by tax class

colors = plt.cm.get_cmap('tab20c', len(pivot))

ax = pivot.plot(kind='bar', legend=False)

for idx, bar in enumerate(ax.patches):
    bar.set_color(colors(idx))

plt.title("Median Sale Price by Tax Class")
plt.xlabel("Tax Class")
plt.ylabel("Median Sale Price")
plt.xticks(rotation=0)
plt.show()


# In[56]:


# Identify and delete variables that have nothing to do with the dependent variable

df.corr().abs().unstack().sort_values(ascending =False )["SALE PRICE"]


# #### Normalizing the data

# In[58]:


# Select numeric columns

numeric_data = df.select_dtypes(include=[np.number])
numeric_data.describe()


# In[59]:


# Transform the numeric features using log(x + 1)

skewed = df[numeric_data.columns].apply(lambda x: skew(x.dropna().astype(float)))
skewed = skewed[skewed > 0.75]
skewed = skewed.index
df[skewed] = np.log1p(df[skewed])


# In[60]:


scaler = StandardScaler()
scaler.fit(df[numeric_data.columns])
scaled = scaler.transform(df[numeric_data.columns])

for i, col in enumerate(numeric_data.columns):
       df[col] = scaled[:,i]
        
df.head()


# #### One hot encoding for categorical columns

# In[63]:


# Drop columns that are not required
del df['BUILDING CLASS AT PRESENT']
del df['BUILDING CLASS AT TIME OF SALE']
del df['NEIGHBORHOOD']
del df['ADDRESS']
del df['SALE MONTH']
del df['SALE YEAR']
del df['BUILDING AGE']


# In[64]:


# Select categorical columns to be one hot encoded

one_hot_variables = ['BOROUGH', 'BUILDING CLASS CATEGORY','TAX CLASS AT PRESENT','TAX CLASS AT TIME OF SALE']
df = pd.get_dummies(df, columns=one_hot_variables, prefix=one_hot_variables, prefix_sep="_" , drop_first=True)
df.info()


# In[65]:


# Test/train split

y = df['SALE PRICE']
X = df.drop('SALE PRICE', axis=1)

X.shape, y.shape


# In[66]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('Train set:')
print(X_train.shape , y_train.shape)
print('Test set')
print(X_test.shape , y_test.shape)


# #### Modeling

# In[68]:


# Define function to calculate RMSE of models

def rmse(y_test,y_pred):
      return np.sqrt(mean_squared_error(y_test,y_pred))


# In[69]:


# Linear Regression

reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)
rmse(y_test,y_pred_reg)


# In[70]:


# Lasso regression

alpha=0.00099
lasso_regr=Lasso(alpha=alpha,max_iter=50000)
lasso_regr.fit(X_train, y_train)
y_pred_lasso=lasso_regr.predict(X_test)
rmse(y_test,y_pred_lasso)


# In[71]:


# Random Forest

rf_regr = RandomForestRegressor()
rf_regr.fit(X_train, y_train)
y_pred_rf = rf_regr.predict(X_test)
rmse(y_test,y_pred_rf)


# In[75]:


# Plot variable importance from random forest model

rankings = rf_regr.feature_importances_.tolist()
importance = pd.DataFrame(sorted(zip(X_train.columns,rankings),reverse=True),columns=["variable","importance"]).sort_values("importance",ascending = False)


# In[79]:


top_10_importance = importance.head(10)

plt.figure(figsize=(10,6))
sns.barplot(x="importance",
            y="variable",
            data=top_10_importance)
plt.title('Variable Importance')
plt.tight_layout()


# In[ ]:




