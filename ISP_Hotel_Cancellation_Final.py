#!/usr/bin/env python


# <h1 ><span style="color:blue">Capstone Project - Hotel Cancellations</span></h1>

# ***INDEX***
# 1. [Problem Statement](#Problem-Statement)
# 2. [Importing Libraries](#Importing-Libraries)
# 3. [Data loading](#Data-Loading)
# 4. [Data Information](#Data-Information)
# 5. [Exploratory Data Analysis](#Exploratory-Data-Analysis)
# 6. [Data Reloading](#Data-Reloading)
# 7. [Model 1](#Model-1:-Logistic-regression)
# 8. [Model 2](#Model-2:-Linear-discriminant-Analysis)
# 9.[Model 3](#Model-3:-KNN-model)
# 10.[Model 4](#Model-4:-Naive-Bayes)
# 11.[Model 5](#Model-5:-Decision-Tree)
# 12.[Model 6](#Model-6:-Linear-Support-Vector-Classifier)
# 13.[Model 7](#Model-7:-Artificial-Neural-Network)
# 14.[Model 8](#Model-8:-Random-Forest)
# 15.[Model 9](#Model-9:-Bagging-Classifier)
# 16.[Model 10](#Model-10:-Gradient-Boosting-Classifier)
# 17.[Tuned Model 1](#Model-Tuning)
# 18.[Feature Importance](#Feature-Importance)

# ## Problem Statement

# This dataset consists of guest booking information of one of the world’s major leading chain of hotels, homes and spaces. The information given should be used to build some predictive models to classify whether a hotel booking is likely to be canceled, which can affect the revenue stream and help in planning the overbooking levels.

# ### Importing Libraries

# importing standard libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics 
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_curve,RocCurveDisplay, plot_confusion_matrix, confusion_matrix,classification_report, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
import joblib


from sklearn.impute import SimpleImputer
pd.set_option('display.max_columns', None)

import datetime as dt

import warnings 
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder




# pip install sidetable
# import sidetable as stb


# ### Data Loading

# In[3]:


# Load the data from the Excel file
file_path = 'OriginalDataset_hotel_revenue_historical_full-2.xlsx'
data_2018 = pd.read_excel(file_path, sheet_name='2018')
data_2019 = pd.read_excel(file_path, sheet_name='2019')
data_2020 = pd.read_excel(file_path, sheet_name='2020')

# Combine data from all three years into a single dataframe
df = pd.concat([data_2018, data_2019, data_2020], ignore_index=True)
df.head()


# In[4]:


# Merge the arrival date columns into a new column 'arrival_date'
df['arrival_date'] = pd.to_datetime(df[['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month']].astype(str).agg('-'.join, axis=1))

# Calculate the booking date
df['booking_date'] = df['arrival_date'] - pd.to_timedelta(df['lead_time'], unit='d')

# Drop the original arrival date columns
df.drop(columns=['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month', 'arrival_date_week_number', 'lead_time'], inplace=True)

# revised df
df.head()


# ### Data Information

# In[5]:


df.info()


# In[6]:


df.isna().sum()


# In[7]:


values = {"agent": "No_Agent", "company": "No_Company"}
df[['agent', 'company']]=df[['agent', 'company']].fillna(value = values)
df[['agent', 'company']]


# In[8]:


#convert some numerical to categorical column names of data frame in a list
Cat_col_names = list(df[['agent', 'company','is_canceled', 'is_repeated_guest']])
print("\nNames of columns to be converted into categorical")
print(Cat_col_names)


# In[9]:


# loop to change each column to category type
for col in Cat_col_names:
    df[col] = df[col].astype(str)

# using apply method
df[['booking_date', 'arrival_date']] = df[['booking_date', 'arrival_date']].apply(pd.to_datetime)


df.info()


# In[10]:


print("First booking Date recorded is : {}." .format(df['booking_date'].min().strftime('%d-%m-%Y')))
print("Last booking Date recorded is : {}." .format(df['booking_date'].max().strftime('%d-%m-%Y')))


# In[11]:


print("First arrival Date recorded is : {}." .format(df['arrival_date'].min().strftime('%d-%m-%Y')))
print("Last arrival Date recorded is : {}." .format(df['arrival_date'].max().strftime('%d-%m-%Y')))


# In[12]:


### imputing the country columns 
# the country column we will fill it with the most frequent value of the column 

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent',verbose=0)
imputer= imputer.fit(df[['country']].iloc[:,:])

df['country']=imputer.transform(df[['country']])
df.isna().sum()


# In[13]:


df['is_canceled'].value_counts(normalize=True)

#The data is imbalanced therefore the accuracy score along with f1 score for cancelled class need to be considered as evaluation matrix score 


# In[14]:


df.describe().T


# there's 0 adult and 0 children in the data as well we will have to do a deeper analysis on this with the business.
# as it is impossible to have a booking with 0 adult and 0 children we will be dropping this anomaly. 
# Company and agent columns have unique ID's which are considered as numerical.


# In[15]:


df.describe(include = 'O').T

# almost half of the booking were made from portugal 
# in this case type_1 hotel is more favored compared to type_2 hotel
# there is a discrepancy of reserved room type and assigned room type 
# hotel might have room that's not sold to the public 
# no deposit here might have a reason why cancellation is high (since customer will not have anything to lose in this no deposit case)
# most popular room is room type A (we need to check it with the ADR which room type has the average daily rate and what it has to do with the popularity )


# In[16]:


df['guests'] = df['adults'] + df['children'] + df['babies']
df.head()

# total guest of the hotel booking by adding the babies children and babies


# In[17]:


# since it's impossible to have 0 guests to book a hotel room, unless they are pseudo rooms reserved by the hotel itself
df[df['guests'] == 0]


# In[18]:


# we will be dropping such rows as they will not be usefule for our analysis.

df.drop(labels = df[df['guests'] == 0].index, axis = 0, inplace = True)
df[df['guests'] == 0]


# In[19]:


#create a new column with total stay nights including both weekend and weekday.

df['total_stays'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
df.head()


# In[20]:


df[df['total_stays'] == 0]

# It has been observed that certain reservations (645 rows) have 0 total stay nights which implies arrival date 
# and departure date is same and quite possible for business meetings. Hence we will not drop such rows.


# #### Lead Time Calculation
# 
# Since the're are many unique values in the lead time we will group it into months before we do analysis on those so hopefully we can see a trend from binning lead time into months

# In[21]:


df['arrival_date'] = df['arrival_date'].dt.strftime('%d-%m-%Y')
df['booking_date'] = df['booking_date'].dt.strftime('%d-%m-%Y')


# In[22]:


df['lead_time'] = (pd.to_datetime(df['arrival_date']) - pd.to_datetime(df['booking_date'])).dt.days
df.head()


# In[23]:


df['lead_time'].describe()


# In[24]:


# stays in weekend nights and stays in week nights now combined into total stays
# Similarly adults and children are combined into guests column 

# we will be dropping four columns 

df = df.drop(columns = ['stays_in_weekend_nights', 'stays_in_week_nights', 'adults','children', 'babies'])
df.head()


# In[25]:


df.duplicated().sum()


# In[26]:


df.skew()


# ### Exploratory Data Analysis

# In[27]:


df.head()


# In[28]:


df.info()


# In[29]:


df.describe().T


# there's 0 adult and 0 children in the data as well we will have to do a deeper analysis on this with the business.
# as it is impossible to have a booking with 0 adult and 0 children we will be dropping this anomaly. 
# Company and agent columns have unique ID's which are considered as numerical.


# In[30]:


df.describe(include = 'O').T

# almost half of the booking were made from portugal 
# in this case type_1 hotel is more favored compared to type_2 hotel
# there is a discrepancy of reserved room type and assigned room type 
# hotel might have room that's not sold to the public 
# no deposit here might have a reason why cancellation is high (since customer will not have anything to lose in this no deposit case)
# most popular room is room type A (we need to check it with the ADR which room type has the average daily rate and what it has to do with the popularity )


# 

# In[31]:


# df.stb.freq(['hotel'], cum_cols=False)

# as we mentioned before that there are more booking from the type_1 hotel compared to the type_2 hotel in this case 
# we will se it later on how this affect cancellation 


# In[32]:


# df.stb.freq(['is_canceled'], cum_cols = False)


# this is the problem that hospitality industy is facing there are almost 4 cancellation in every 10 bookings 

# this data is  almost balance, so later on for the machine learning process we wont need to do an imbalance handling 


# In[33]:


# df.stb.freq(['lead_time'], cum_cols = False)


# In[34]:


# df.stb.freq(['distribution_channel'], cum_cols= False)

# as we can see from the table below that Travel Agent or Tour Operator is the biggest booking distribution channel 
# compared to other channel while direct is the second most biggest distribution channel

# we will treat the undefined values as TA / TO 


# In[35]:


df['distribution_channel'] =  df['distribution_channel'].str.replace('Undefined', 'TA/TO')


# In[36]:


# df.stb.freq(['distribution_channel'], cum_cols= False)


# In[37]:


# df.stb.freq(['market_segment'], cum_cols = False)

# the market segment is almost similar to the distribution channel 
# however we see that there are more categories in this column compared to the distribution channel 
# we see from here that travel agent (online and offline) market segment dominating the booking compared to other market segment

# there are aa couple of undefined values as well we will replace it with mode here in this case is 'Online TA'
# this is because of imputing the random value with the most frequent value in the column


# In[38]:


df['market_segment'] = df['market_segment'].replace(['Undefined', 'Online Travel Agents'], 'Online TA')
# df.stb.freq(['market_segment'], cum_cols = False)

# how we have replace the undefined value in the column


# In[39]:


# df.stb.freq(['meal'], cum_cols=False)

# bed and breakfast is the most popular meal package compared to the rest of the meal package 
# while full board is the least popular meal package compared to the rest of the meal package (included breakfast, lunch , dinner )


# In[40]:


# df.stb.freq(['country'], cum_cols = False).head(20)

# almost half of the booking is made from portugal (this is kind of make sense since the both of the hotel is in portugal)

# since that there's so many unique values from all the countries where the booking comes from we will try to group it into continent
# or we will group it into booking from portugal and booking from outside portugal since both of the hotel are in portugal
# it's kind of make sense to split the booking into international booking or local booking 


# In[41]:


# df.stb.freq(['reserved_room_type'], cum_cols = False)

# as from our df.describe(include = 'O') we saw that room A is the most popular room in the bookings 
# we will figure out later why, our assumption for now is room A is the cheapest room in the booking 
# compared to any other rooms


# In[42]:


def assigned(row):
    if (row['reserved_room_type'] == row['assigned_room_type']):
        return "Same"
    else :
        return "Different"

df['assigned_room_type'] = df.apply(assigned, axis = 1)
# df.stb.freq(['assigned_room_type'], cum_cols = False)

# Around 12.4% of reservations had assigned room type different from reserved room type


# In[43]:


# df.stb.freq(['deposit_type'], cum_cols = False)

# as we mentioned before that the No deposit type is the most popular compared to other deposit type in this booking in portugal 
# this might be the reason why the cancellation in the industry has been on a rise
# the flexibility that's given to the customer to book hotel without any deposit


# In[44]:


# df.stb.freq(['is_repeated_guest'], cum_cols = False)

# from this table below we see that there are only 3.1 % of repeated guest from this booking 
# this might be affecting the cancellation of the hotel from the low rate of loyal customers 
# we will deep dive into this later on 

# since we will not solely looking into loyal customer but this is few reason why loyal customer are more profitable than
# the old ones 
# 1. They already know your value.
# 2. They cost less to service.
# 3. They refer more business.
# 4. They will buy and pay more

# source : http://www.converoinc.com/4-reasons-existing-customers-are-more-profitable-than-new-ones/


# In[45]:


# df.stb.freq(['previous_cancellations'], cum_cols = False)

# almost 95% of the booking never been cancelled before in this data set
# we will group this into booking that's never been cancelled or have been cancelled before 


# In[46]:


def cancellation(row):
    if (row['previous_cancellations'] == 0):
        return 0
    else :
        return 1

df['is_previously_cancelled'] = df.apply(cancellation, axis = 1)
# df.stb.freq(['is_previously_cancelled'], cum_cols = False)


# In[47]:


# df.stb.freq(['booking_changes'], cum_cols = False)

# almost 85 % of the customers never change their booking 
# since there are many values of this booking changes column 
# we will group it in to does the booking ever been changes or not 


# In[48]:


def changes(row):
    if (row['booking_changes'] == 0):
        return 0
    else :
        return 1

df['is_booking_changes'] = df.apply(changes, axis = 1)
# df.stb.freq(['is_booking_changes'], cum_cols = False)


# In[49]:


# booking_changes and previous_cancellations column is transformed 
# into boolean columns is_booking_changes and is_previously_cancelled


# we will be dropping old columns

df = df.drop(columns = ['booking_changes','previous_cancellations'])
df.head()


# In[50]:


# df.stb.freq(['total_of_special_requests'], cum_cols = False)

# more than half of the customers don't have any special request when they book
# we will see later if special request has an effect on the cancellation


# In[51]:


# df.stb.freq(['customer_type'], cum_cols = False)


# majority of the booking  customer here are transient (individual booking /personal not related to company or anything )
# we will see how this customer type affecting the cancellation rate as well 


# In[52]:


cat=[]
num=[]
for i in df.columns:
    if df[i].dtype=="object":
        cat.append(i)
    else:
        num.append(i)

print("Categorical columns are :", cat,"\n") 
print("Numerical columns are :", num)


# In[53]:


# Drop date columns from numerical columns
num= [e for e in num if e not in ('reservation_status_date')]


### since the there is no null value anymore in the data now i will check the outliers 

plt.figure(figsize = (20, 20))
x = 1 

for column in num[::]:
    plt.subplot(5,2,x)
    sns.boxplot(df[column]).set_title(column)
    x+=1
    
plt.tight_layout()
plt.suptitle('Boxplots of Numerical Data', va='bottom')
#plt.show()

# we can see that there are huge outliers in many of the columns 
# we will handle the outliers by binning the columns that has outliers in it and from the box plot there are columns that has a random value like 0 number of adults 


# In[54]:


def remove_outlier(col):
    sorted(col)
    Q1,Q3=np.percentile(col,[25,75])
    IQR=Q3-Q1
    lower_range= Q1-(1.5 * IQR)
    upper_range= Q3+(1.5 * IQR)
    return lower_range, upper_range


# In[55]:


lr,ur=remove_outlier(df[num])
df[num]=np.where(df[num]>ur,ur,df[num])
df[num]=np.where(df[num]<lr,lr,df[num])


# In[56]:


plt.figure(figsize = (20, 10))
x = 1 

for column in num[::]:
    plt.subplot(6,3,x)
    sns.boxplot(df[column])
    x+=1
    
plt.tight_layout()


# In[57]:


plt.rcParams['figure.figsize']=15,10
plt.subplot(121)
plt.pie(df['is_canceled'].value_counts().values,
        labels=df['is_canceled'].value_counts().index,
        startangle=90,
        colors=['red', 'blue'],
        explode=[0.05,0.05],
        shadow=True, autopct='%1.2f%%')
plt.subplot(122)
plt.pie(df['hotel'].value_counts().values,
        labels=df['hotel'].value_counts().index,
        startangle=90,
        colors=['pink', 'lightblue'],
        explode=[0.05,0.05],
        shadow=True, autopct='%1.2f%%')

plt.suptitle('Distribution of Cancellations and Hotel Type')

# plt.savefig('plots/categorical.png')
#plt.show()


# In[58]:


plt.rcParams['figure.figsize']=15,10
plt.subplot(121)
plt.pie(df['deposit_type'].value_counts().values,
        labels=df['deposit_type'].value_counts().index,
        startangle=90,
        colors=['red', 'blue', 'green'],
        explode=[0.05,0.05,0.7],
        shadow=True, autopct='%1.2f%%')
plt.subplot(122)
plt.pie(df['customer_type'].value_counts().values,
        labels=df['customer_type'].value_counts().index,
        startangle=90,
        colors=['pink', 'lightblue', 'brown', 'yellow'],
        explode=[0.05,0.05,0.1,0.7],
        shadow=True, autopct='%1.2f%%')

plt.suptitle('Distribution of deposit type and customer Type')

# plt.savefig('plots/categorical.png')
#plt.show()


# In[59]:


distribution_channel = df.stb.freq(['distribution_channel'], cum_cols = False)
market_segment = df.stb.freq(['market_segment'], cum_cols = False)
market_segment

plt.figure(figsize = (10, 10))

plt.subplot(1,1,1)
market_segment['percent'].plot.pie(explode = [0, 0.2, 0.2, 0.2, 0.2, 0.7, 1.5], 
                                               autopct = '%1.2f%%',
                                               shadow = True
                                               )
plt.legend(market_segment['market_segment'],loc='upper left')
plt.title('Market Segment')
#plt.show()

plt.figure(figsize = (15, 15))
plt.subplot(2,1,1)
distribution_channel['percent'].plot.pie(explode = [0, 0.2, 0.2, 0.7], 
                                               autopct = '%1.2f%%',
                                               shadow = True
                                               )
plt.legend(distribution_channel['distribution_channel'])
plt.title('distribution Channel')

plt.tight_layout()
#plt.show()


# In[60]:


plt.rcParams['figure.figsize']=15,10
plt.subplot(121)
plt.pie(df['reserved_room_type'].value_counts().values,
        labels=df['reserved_room_type'].value_counts().index,
        startangle=90,
        colors=['red', 'blue', 'green', 'yellow', 'pink', 'brown', 'green', 'orange', 'cyan'],
        explode=[0.01,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11],
        shadow=True, autopct='%1.2f%%')
plt.subplot(122)
plt.pie(df['assigned_room_type'].value_counts().values,
        labels=df['assigned_room_type'].value_counts().index,
        startangle=90,
        colors=['pink', 'lightblue'],
        explode=[0.05,0.05],
        shadow=True, autopct='%1.2f%%')

plt.suptitle('Distribution of reserved and assigned room Type')

# plt.savefig('plots/categorical.png')
#plt.show()


# In[61]:


plt.figure(figsize = (20, 20))
x = 1 

for column in df.describe().columns:
    plt.subplot(8,3,x)
    sns.boxplot(df[column])
    x+=1
    
plt.tight_layout()

# as we can see that there are still many outliers in many of the columns 
# we are not going to drop or treat the outliers here as outliers could provide many useful information 
# we can bin in categories and create new column and hopefully we will be able to extract some more information by doing that 


# In[62]:


plt.figure(figsize = (20, 20))
x = 1 

for column in df.describe().columns:
    plt.subplot(8,4,x)
    sns.distplot(df[column])
    x+=1
    
plt.tight_layout()


# In[63]:


print(num)


# In[64]:


# plotting the canceled variable with other numerical variables
for x in num:
    plt.figure(figsize=(8,5))
    sns.stripplot(x=df["is_canceled"], y=df[x])
    #plt.show()


# In[65]:


cat_1 = ['hotel', 'meal', 'market_segment',
         'distribution_channel', 'is_repeated_guest', 
         'deposit_type', 'customer_type']
# plotting the canceled variable with other categorical variables
for x in cat_1[::]:
    plt.figure(figsize=(20,10))
    sns.countplot(hue=df["is_canceled"], x=df[x])
    #plt.show()


# In[66]:


# plotting the canceled variable with lead time with respect to hotel type

plt.figure(figsize=(8,5))
sns.catplot(x="hotel", y="lead_time", hue="is_canceled", kind="bar", data = df)
#plt.show();


# In[67]:


# sns.pairplot(df,hue='is_canceled')


# In[68]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
#plt.show()


# In[69]:


#since the data has been cleaned it has no more missing value and some randomness to the data we will export this data 
# Can do a tableau visualization from the data.

df.to_excel('hotel_cleaned.xlsx' ,index= False)


# In[70]:


#No. of duplicate records after data cleaning
print('No. of duplicate rows in the data :',df.duplicated().sum())


# In[71]:


# after dropping duplicate rows
df.drop_duplicates(keep=False,inplace=True)
print('No. of duplicate rows in the data after dropping :',df.duplicated().sum())


# In[72]:


# prior to scaling
plt.figure(figsize=(8,8))
plt.plot(df[num])
plt.legend(df[num], loc='upper right')
#plt.show()


# In[73]:


# Creating an object for the StandardScaler function
X = StandardScaler()

# fitting and transforming the numerical columns 
df['days_in_waiting_list'] = X.fit_transform(df[['days_in_waiting_list']])
df['required_car_parking_spaces'] = X.fit_transform(df[['required_car_parking_spaces']])
df['totalno_of_special_requests'] = X.fit_transform(df[['total_of_special_requests']])
df['guests'] = X.fit_transform(df[['guests']])
df['total_stays'] = X.fit_transform(df[['total_stays']])
df['lead_time'] = X.fit_transform(df[['lead_time']])

    

#after scaling plot visualization
plt.figure(figsize=(8,8))
plt.plot(df[num])
plt.legend(df[num])
#plt.show()


# In[74]:


# we will be dropping agent and company columns 
df = df.drop(columns = ['agent', 'company', 'previous_bookings_not_canceled', 'is_previously_cancelled', 'reservation_status'])
df.head()


# In[75]:


def changes(row):
    if (row['country'] == 'PRT'):
        return 'Domestic'
    else :
        return 'International'

df['country'] = df.apply(changes, axis = 1)
# df.stb.freq(['country'], cum_cols = False)


# In[76]:


def changes(row):
    if (row['reserved_room_type'] == 'A'):
        return 'Standard'
    else :
        return 'Premium'

df['reserved_room_type'] = df.apply(changes, axis = 1)
# df.stb.freq(['reserved_room_type'], cum_cols = False)


# In[77]:


# creating dummy variables for ML model
model_df = pd.get_dummies(df, prefix=['hotel',
                             'meal',
                             'market_segment',
                             'distribution_channel',
                             'reserved_room_type',
                             'assigned_room_type',
                             'deposit_type',
                             'customer_type'], 
                    columns= ['hotel',
                             'meal',
                             'market_segment',
                             'distribution_channel',
                             'reserved_room_type',
                             'assigned_room_type',
                             'deposit_type',
                             'customer_type'],
                  drop_first = True)
model_df.head()


# In[78]:


# model_df.to_excel('hotel_model_data.xlsx' ,index= False)


# ### Data Reloading

# In[2]:


df = pd.read_excel('hotel_cleaned.xlsx')
df.head()


# ### Data Information

# In[3]:


df.info()


# In[4]:


df.isna().sum()


# In[5]:


#No. of duplicate records after data cleaning
print('No. of duplicate rows in the data :',df.duplicated().sum())


# In[6]:


# after dropping duplicate rows
df.drop_duplicates(keep=False,inplace=True)
print('No. of duplicate rows in the data after dropping :',df.duplicated().sum())


# In[7]:


df.info()


# In[8]:


# Numerical column description
df.describe().T


# In[9]:


# Categorical column description
df.describe(include='O').T


# In[10]:


#country column
# df.stb.freq(['country'], cum_cols = False).head(10)

# since maximum share is from top 5 countries portugal, great britain, France, Spain and Germany, we will be grouping rest of the 170 countries into others.


# In[11]:


#Categorising countries into six categories : 

def changes(row):
    if (row['country'] == 'PRT'):
        return 'Portugal'
    elif (row['country'] == 'GBR'):
        return 'Great_Britain'
    elif (row['country'] == 'FRA'):
        return 'France'
    elif (row['country'] == 'ESP'):
        return 'Spain'
    elif (row['country'] == 'DEU'):
        return 'Germany'
    else :
        return 'Others'

df['country'] = df.apply(changes, axis = 1)
# df.stb.freq(['country'], cum_cols = False)


# In[12]:


# df.stb.freq(['reserved_room_type'], cum_cols = False)

# as from our df.describe(include = 'O') we saw that room A is the most popular room in the bookings 
# we will figure out later why, our assumption for now is room A is the cheapest room in the booking 
# compared to any other rooms


# In[13]:


def changes(row):
    if (row['reserved_room_type'] == 'A'):
        return 'Standard'
    else :
        return 'Premium'

df['reserved_room_type'] = df.apply(changes, axis = 1)
df.stb.freq(['reserved_room_type'], cum_cols = False)


# In[14]:


# creating dummy variables for ML model
model_df = pd.get_dummies(df, prefix=['hotel',
                             'meal',
                             'market_segment',
                             'distribution_channel',
                             'reserved_room_type',
                             'assigned_room_type',
                             'deposit_type',
                             'customer_type',
                             'country'], 
                    columns= ['hotel',
                             'meal',
                             'market_segment',
                             'distribution_channel',
                             'reserved_room_type',
                             'assigned_room_type',
                             'deposit_type',
                             'customer_type',
                             'country'],
                  drop_first = True)
model_df.head()


# In[15]:


model_df = model_df.drop(['booking_date','arrival_date','agent','company',
                          'previous_bookings_not_canceled', 'is_previously_cancelled', 
                          'reservation_status', 'reservation_status_date'], axis=1)


# In[16]:


model_df.info()


# In[17]:


model_df['adr'].where(model_df['adr'] >= 1, 1, inplace=True)

model_df['adr'].fillna(0).astype(np.int64, errors='ignore')


model_df['adr'].describe()


# In[18]:


model_df['guests'].where(model_df['guests'] >= 1, 1, inplace=True)

model_df['guests'].dropna().astype(np.int64, errors='ignore')


model_df['adr'].describe()


# In[19]:


model_df.dropna(subset=['guests'])
model_df.info()


# In[20]:


# Copy all the predictor variables into X dataframe
X = model_df.drop('is_canceled', axis=1)

# Copy target into the y dataframe. 
y = model_df[['is_canceled']]


# In[21]:


# Split X and y into training and test set in 70:30 ratio
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30 , random_state=1, stratify = model_df['is_canceled'])


# In[22]:


print (y_train.value_counts(normalize = True).round(4))
print (y_test.value_counts(normalize = True).round(4))


# ### Model 1: Logistic regression

# In[23]:


X_train.head()


# In[24]:


# invoke the LogisticRegression function and find the bestfit model on training data
LR_model = LogisticRegression()
LR_model.fit(X_train, y_train)
LR_model.get_params()


# In[25]:


## Performance Matrix on train data set
ytrain_predict = LR_model.predict(X_train)
model_score = LR_model.score(X_train, y_train)
print('\n \033[1m Train Accuracy Score : \033[0m', model_score)
print('\n \033[1m Train Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_train, ytrain_predict))
print('\n \033[1m Train Classification Report : \033[0m \n', metrics.classification_report(y_train, ytrain_predict))


# In[26]:


## Performance Matrix on test data set
ytest_predict = LR_model.predict(X_test)
test_model_score = LR_model.score(X_test, y_test)
print('\n \033[1m Test Accuracy Score : \033[0m', test_model_score)
print('\n \033[1m Test Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_test, ytest_predict))
print('\n \033[1m Test Classification Report : \033[0m \n', metrics.classification_report(y_test, ytest_predict))


# In[27]:


f,a =  plt.subplots(1,2,sharex=True,sharey=True,squeeze=False, figsize=(10, 5))

#Plotting confusion matrix for the Training Data and Test data

plot_0 = sns.heatmap((metrics.confusion_matrix(y_train, ytrain_predict)),annot=True,fmt='.5g',cmap='GnBu',ax=a[0][0]);
a[0][0].set_title('Training Data');

plot_1 = sns.heatmap((metrics.confusion_matrix(y_test, ytest_predict)),annot=True,fmt='.5g',cmap='OrRd',ax=a[0][1]);
a[0][1].set_title('Test Data');

plt.suptitle('Logistic Regression');

plt.savefig('plots/LR_CM.png')


# In[28]:


# AUC and ROC for the training and testing data
# Training Data Probability Prediction
pred_prob_train = LR_model.predict_proba(X_train)

# Test Data Probability Prediction
pred_prob_test = LR_model.predict_proba(X_test)

# AUC and ROC for the train data

# calculate AUC
auc = metrics.roc_auc_score(y_train,pred_prob_train[:,1])
print('AUC for the Training Data: %.3f' % auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_train,pred_prob_train[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label = 'Logit')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Training Data')


# AUC and ROC for the test data

# calculate AUC
test_auc = metrics.roc_auc_score(y_test,pred_prob_test[:,1])
print('AUC for the Test Data: %.3f' % test_auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_test,pred_prob_test[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label= 'No skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label='Logit')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Test data')

# save and show the plot
plt.savefig('plots/LR_ROC.png')
#plt.show()


# ### Model 2: Linear discriminant Analysis




# invoke the LogisticRegression function and find the bestfit model on training data
LDA_model = LinearDiscriminantAnalysis()
LDA_model.fit(X_train, y_train)
LDA_model.get_params()


# In[30]:


## Performance Matrix on train data set
ytrain_predict = LDA_model.predict(X_train)
model_score = LDA_model.score(X_train, y_train)
print('\n \033[1m Train Accuracy Score : \033[0m', model_score)
print('\n \033[1m Train Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_train, ytrain_predict))
print('\n \033[1m Train Classification Report : \033[0m \n', metrics.classification_report(y_train, ytrain_predict))


# In[31]:


## Performance Matrix on test data set
ytest_predict = LDA_model.predict(X_test)
test_model_score = LDA_model.score(X_test, y_test)
print('\n \033[1m Test Accuracy Score : \033[0m', test_model_score)
print('\n \033[1m Test Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_test, ytest_predict))
print('\n \033[1m Test Classification Report : \033[0m \n', metrics.classification_report(y_test, ytest_predict))


# In[32]:


f,a =  plt.subplots(1,2,sharex=True,sharey=True,squeeze=False, figsize=(10, 5))

#Plotting confusion matrix for the Training Data and Test data

plot_0 = sns.heatmap((metrics.confusion_matrix(y_train, ytrain_predict)),annot=True,fmt='.5g',cmap='GnBu',ax=a[0][0]);
a[0][0].set_title('Training Data');

plot_1 = sns.heatmap((metrics.confusion_matrix(y_test, ytest_predict)),annot=True,fmt='.5g',cmap='OrRd',ax=a[0][1]);
a[0][1].set_title('Test Data');

plt.suptitle('LDA Model');

plt.savefig('plots/LDA_CM.png')


# In[33]:


# AUC and ROC for the training and testing data
# Training Data Probability Prediction
pred_prob_train = LDA_model.predict_proba(X_train)

# Test Data Probability Prediction
pred_prob_test = LDA_model.predict_proba(X_test)

# AUC and ROC for the train data

# calculate AUC
auc = metrics.roc_auc_score(y_train,pred_prob_train[:,1])
print('AUC for the Training Data: %.3f' % auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_train,pred_prob_train[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label = 'LDA')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Training Data')


# AUC and ROC for the test data

# calculate AUC
test_auc = metrics.roc_auc_score(y_test,pred_prob_test[:,1])
print('AUC for the Test Data: %.3f' % test_auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_test,pred_prob_test[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label= 'No skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label='LDA')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Test data')

# save and show the plot
plt.savefig('plots/LDA_ROC.png')
#plt.show()


# ### Model 3: KNN model

# In[34]:


# invoke the LogisticRegression function and find the bestfit model on training data
KNN_model = KNeighborsClassifier()
KNN_model.fit(X_train, y_train)
KNN_model.get_params()


# In[ ]:


## Performance Matrix on train data set
ytrain_predict = KNN_model.predict(X_train)
model_score = KNN_model.score(X_train, y_train)
print('\n \033[1m Train Accuracy Score : \033[0m', model_score)
print('\n \033[1m Train Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_train, ytrain_predict))
print('\n \033[1m Train Classification Report : \033[0m \n', metrics.classification_report(y_train, ytrain_predict))


# In[36]:


## Performance Matrix on test data set
ytest_predict = KNN_model.predict(X_test)
test_model_score = KNN_model.score(X_test, y_test)
print('\n \033[1m Test Accuracy Score : \033[0m', test_model_score)
print('\n \033[1m Test Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_test, ytest_predict))
print('\n \033[1m Test Classification Report : \033[0m \n', metrics.classification_report(y_test, ytest_predict))


# In[37]:


f,a =  plt.subplots(1,2,sharex=True,sharey=True,squeeze=False, figsize=(10, 5))

#Plotting confusion matrix for the Training Data and Test data

plot_0 = sns.heatmap((metrics.confusion_matrix(y_train, ytrain_predict)),annot=True,fmt='.5g',cmap='GnBu',ax=a[0][0]);
a[0][0].set_title('Training Data');

plot_1 = sns.heatmap((metrics.confusion_matrix(y_test, ytest_predict)),annot=True,fmt='.5g',cmap='OrRd',ax=a[0][1]);
a[0][1].set_title('Test Data');

plt.suptitle('KNN Model');

plt.savefig('plots/KNN_CM.png')



# AUC and ROC for the training and testing data
# Training Data Probability Prediction
pred_prob_train = KNN_model.predict_proba(X_train)

# Test Data Probability Prediction
pred_prob_test = KNN_model.predict_proba(X_test)

# AUC and ROC for the train data

# calculate AUC
auc = metrics.roc_auc_score(y_train,pred_prob_train[:,1])
print('AUC for the Training Data: %.3f' % auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_train,pred_prob_train[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label = 'KNN')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Training Data')


# AUC and ROC for the test data

# calculate AUC
test_auc = metrics.roc_auc_score(y_test,pred_prob_test[:,1])
print('AUC for the Test Data: %.3f' % test_auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_test,pred_prob_test[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label= 'No skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label='KNN')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Test data')

# save and show the plot
plt.savefig('plots/KNN_ROC.png')
#plt.show()


# ### Model 4: Naive Bayes 



# invoke the LogisticRegression function and find the bestfit model on training data
NB_model = GaussianNB()
NB_model.fit(X_train, y_train)
NB_model.get_params()




## Performance Matrix on train data set
ytrain_predict = NB_model.predict(X_train)
model_score = NB_model.score(X_train, y_train)
print('\n \033[1m Train Accuracy Score : \033[0m', model_score)
print('\n \033[1m Train Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_train, ytrain_predict))
print('\n \033[1m Train Classification Report : \033[0m \n', metrics.classification_report(y_train, ytrain_predict))




## Performance Matrix on test data set
ytest_predict = NB_model.predict(X_test)
test_model_score = NB_model.score(X_test, y_test)
print('\n \033[1m Test Accuracy Score : \033[0m', test_model_score)
print('\n \033[1m Test Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_test, ytest_predict))
print('\n \033[1m Test Classification Report : \033[0m \n', metrics.classification_report(y_test, ytest_predict))



f,a =  plt.subplots(1,2,sharex=True,sharey=True,squeeze=False, figsize=(10, 5))

#Plotting confusion matrix for the Training Data and Test data

plot_0 = sns.heatmap((metrics.confusion_matrix(y_train, ytrain_predict)),annot=True,fmt='.5g',cmap='GnBu',ax=a[0][0]);
a[0][0].set_title('Training Data');

plot_1 = sns.heatmap((metrics.confusion_matrix(y_test, ytest_predict)),annot=True,fmt='.5g',cmap='OrRd',ax=a[0][1]);
a[0][1].set_title('Test Data');

plt.suptitle('Naive Bayes Model');

plt.savefig('plots/NB_CM.png')



# AUC and ROC for the training and testing data
# Training Data Probability Prediction
pred_prob_train = NB_model.predict_proba(X_train)

# Test Data Probability Prediction
pred_prob_test = NB_model.predict_proba(X_test)

# AUC and ROC for the train data

# calculate AUC
auc = metrics.roc_auc_score(y_train,pred_prob_train[:,1])
print('AUC for the Training Data: %.3f' % auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_train,pred_prob_train[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label = 'NB')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Training Data')


# AUC and ROC for the test data

# calculate AUC
test_auc = metrics.roc_auc_score(y_test,pred_prob_test[:,1])
print('AUC for the Test Data: %.3f' % test_auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_test,pred_prob_test[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label= 'No skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label='NB')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Test data')

# save and show the plot
plt.savefig('plots/NB_ROC.png')
#plt.show()


# ### Model 5: Decision Tree

# In[ ]:


# invoke the LogisticRegression function and find the bestfit model on training data
DT_model = DecisionTreeClassifier()
DT_model.fit(X_train, y_train)
DT_model.get_params()


# In[ ]:


## Performance Matrix on train data set
ytrain_predict = DT_model.predict(X_train)
model_score = DT_model.score(X_train, y_train)
print('\n \033[1m Train Accuracy Score : \033[0m', model_score)
print('\n \033[1m Train Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_train, ytrain_predict))
print('\n \033[1m Train Classification Report : \033[0m \n', metrics.classification_report(y_train, ytrain_predict))


# In[ ]:


## Performance Matrix on test data set
ytest_predict = DT_model.predict(X_test)
test_model_score = DT_model.score(X_test, y_test)
print('\n \033[1m Test Accuracy Score : \033[0m', test_model_score)
print('\n \033[1m Test Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_test, ytest_predict))
print('\n \033[1m Test Classification Report : \033[0m \n', metrics.classification_report(y_test, ytest_predict))


# In[ ]:


f,a =  plt.subplots(1,2,sharex=True,sharey=True,squeeze=False, figsize=(10, 5))

#Plotting confusion matrix for the Training Data and Test data

plot_0 = sns.heatmap((metrics.confusion_matrix(y_train, ytrain_predict)),annot=True,fmt='.5g',cmap='GnBu',ax=a[0][0]);
a[0][0].set_title('Training Data');

plot_1 = sns.heatmap((metrics.confusion_matrix(y_test, ytest_predict)),annot=True,fmt='.5g',cmap='OrRd',ax=a[0][1]);
a[0][1].set_title('Test Data');

plt.suptitle('Decision Tree');

plt.savefig('plots/DT_CM.png')


# In[ ]:


# AUC and ROC for the training and testing data
# Training Data Probability Prediction
pred_prob_train = DT_model.predict_proba(X_train)

# Test Data Probability Prediction
pred_prob_test = DT_model.predict_proba(X_test)

# AUC and ROC for the train data

# calculate AUC
auc = metrics.roc_auc_score(y_train,pred_prob_train[:,1])
print('AUC for the Training Data: %.3f' % auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_train,pred_prob_train[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label = 'DT')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Training Data')


# AUC and ROC for the test data

# calculate AUC
test_auc = metrics.roc_auc_score(y_test,pred_prob_test[:,1])
print('AUC for the Test Data: %.3f' % test_auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_test,pred_prob_test[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label= 'No skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label='DT')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Test data')

# save and show the plot
plt.savefig('plots/DT_ROC.png')
#plt.show()


# ### Model 6: Support Vector Machine

# In[ ]:


# invoke the LogisticRegression function and find the bestfit model on training data
SVC_model = SVC(kernel="linear", probability=True)
SVC_model.fit(X_train, y_train)
SVC_model.get_params()


# In[ ]:


## Performance Matrix on train data set
ytrain_predict = SVC_model.predict(X_train)
model_score = SVC_model.score(X_train, y_train)
print('\n \033[1m Train Accuracy Score : \033[0m', model_score)
print('\n \033[1m Train Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_train, ytrain_predict))
print('\n \033[1m Train Classification Report : \033[0m \n', metrics.classification_report(y_train, ytrain_predict))


# In[ ]:


## Performance Matrix on test data set
ytest_predict = SVC_model.predict(X_test)
test_model_score = SVC_model.score(X_test, y_test)
print('\n \033[1m Test Accuracy Score : \033[0m', test_model_score)
print('\n \033[1m Test Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_test, ytest_predict))
print('\n \033[1m Test Classification Report : \033[0m \n', metrics.classification_report(y_test, ytest_predict))


# In[ ]:


f,a =  plt.subplots(1,2,sharex=True,sharey=True,squeeze=False, figsize=(10, 5))

#Plotting confusion matrix for the Training Data and Test data

plot_0 = sns.heatmap((metrics.confusion_matrix(y_train, ytrain_predict)),annot=True,fmt='.5g',cmap='GnBu',ax=a[0][0]);
a[0][0].set_title('Training Data');

plot_1 = sns.heatmap((metrics.confusion_matrix(y_test, ytest_predict)),annot=True,fmt='.5g',cmap='OrRd',ax=a[0][1]);
a[0][1].set_title('Test Data');

plt.suptitle('Support Vector Machine');

plt.savefig('plots/SVM_CM.png')


# In[ ]:


# AUC and ROC for the training and testing data

# AUC and ROC for the train data

# calculate AUC
decision_scores = SVC_model.decision_function(X_train)
fpr, tpr, thres = roc_curve(y_train, decision_scores)
print('AUC for the Training Data:: {:.3f}'.format(roc_auc_score(y_train, decision_scores)))

#  calculate roc curve
plt.plot(fpr, tpr, "b", label='Linear SVM')
plt.plot([0,1],[0,1], "k--", label='No Skill')
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.legend(loc="best")
plt.title("ROC curve")
#plt.show()
# AUC and ROC for the test data

# calculate AUC
decision_scores = SVC_model.decision_function(X_test)
fpr, tpr, thres = roc_curve(y_test, decision_scores)
print('AUC for the Testing Data:: {:.3f}'.format(roc_auc_score(y_test, decision_scores)))

#  calculate roc curve
plt.plot(fpr, tpr, "b", label='Linear SVM')
plt.plot([0,1],[0,1], "k--", label='No Skill')
plt.xlabel("false positive rate")
plt.ylabel("true positive rate")
plt.legend(loc="best")
plt.title("ROC curve")


# save and show the plot
plt.savefig('plots/SVM_ROC.png')
#plt.show()


# ### Model 7: Artificial Neural Network

# In[ ]:


# invoke the LogisticRegression function and find the bestfit model on training data
ANN_model = MLPClassifier()
ANN_model.fit(X_train, y_train)
ANN_model.get_params()


# In[ ]:


## Performance Matrix on train data set
ytrain_predict = ANN_model.predict(X_train)
model_score = ANN_model.score(X_train, y_train)
print('\n \033[1m Train Accuracy Score : \033[0m', model_score)
print('\n \033[1m Train Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_train, ytrain_predict))
print('\n \033[1m Train Classification Report : \033[0m \n', metrics.classification_report(y_train, ytrain_predict))


# In[ ]:


## Performance Matrix on test data set
ytest_predict = ANN_model.predict(X_test)
test_model_score = ANN_model.score(X_test, y_test)
print('\n \033[1m Test Accuracy Score : \033[0m', test_model_score)
print('\n \033[1m Test Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_test, ytest_predict))
print('\n \033[1m Test Classification Report : \033[0m \n', metrics.classification_report(y_test, ytest_predict))


# In[ ]:


f,a =  plt.subplots(1,2,sharex=True,sharey=True,squeeze=False, figsize=(10, 5))

#Plotting confusion matrix for the Training Data and Test data

plot_0 = sns.heatmap((metrics.confusion_matrix(y_train, ytrain_predict)),annot=True,fmt='.5g',cmap='GnBu',ax=a[0][0]);
a[0][0].set_title('Training Data');

plot_1 = sns.heatmap((metrics.confusion_matrix(y_test, ytest_predict)),annot=True,fmt='.5g',cmap='OrRd',ax=a[0][1]);
a[0][1].set_title('Test Data');

plt.suptitle('Artificial Neural Network model');

plt.savefig('plots/ANN_CM.png')


# In[ ]:


# AUC and ROC for the training and testing data
# Training Data Probability Prediction
pred_prob_train = ANN_model.predict_proba(X_train)

# Test Data Probability Prediction
pred_prob_test = ANN_model.predict_proba(X_test)

# AUC and ROC for the train data

# calculate AUC
auc = metrics.roc_auc_score(y_train,pred_prob_train[:,1])
print('AUC for the Training Data: %.3f' % auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_train,pred_prob_train[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label = 'ANN')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Training Data')


# AUC and ROC for the test data

# calculate AUC
test_auc = metrics.roc_auc_score(y_test,pred_prob_test[:,1])
print('AUC for the Test Data: %.3f' % test_auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_test,pred_prob_test[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label= 'No skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label='ANN')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Test data')

# save and show the plot
plt.savefig('plots/ANN_ROC.png')
#plt.show()


# ### Model 8: Random Forest

# In[ ]:


# invoke the LogisticRegression function and find the bestfit model on training data
RF_model = RandomForestClassifier()
RF_model.fit(X_train, y_train)
RF_model.get_params()


# In[ ]:


## Performance Matrix on train data set
ytrain_predict = RF_model.predict(X_train)
model_score = RF_model.score(X_train, y_train)
print('\n \033[1m Train Accuracy Score : \033[0m', model_score)
print('\n \033[1m Train Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_train, ytrain_predict))
print('\n \033[1m Train Classification Report : \033[0m \n', metrics.classification_report(y_train, ytrain_predict))


# In[ ]:


## Performance Matrix on test data set
ytest_predict = RF_model.predict(X_test)
test_model_score = RF_model.score(X_test, y_test)
print('\n \033[1m Test Accuracy Score : \033[0m', test_model_score)
print('\n \033[1m Test Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_test, ytest_predict))
print('\n \033[1m Test Classification Report : \033[0m \n', metrics.classification_report(y_test, ytest_predict))


# In[ ]:


f,a =  plt.subplots(1,2,sharex=True,sharey=True,squeeze=False, figsize=(10, 5))

#Plotting confusion matrix for the Training Data and Test data

plot_0 = sns.heatmap((metrics.confusion_matrix(y_train, ytrain_predict)),annot=True,fmt='.5g',cmap='GnBu',ax=a[0][0]);
a[0][0].set_title('Training Data');

plot_1 = sns.heatmap((metrics.confusion_matrix(y_test, ytest_predict)),annot=True,fmt='.5g',cmap='OrRd',ax=a[0][1]);
a[0][1].set_title('Test Data');

plt.suptitle('Random Forest');

plt.savefig('plots/RF_CM.png')


# In[ ]:


# AUC and ROC for the training and testing data
# Training Data Probability Prediction
pred_prob_train = RF_model.predict_proba(X_train)

# Test Data Probability Prediction
pred_prob_test = RF_model.predict_proba(X_test)

# AUC and ROC for the train data

# calculate AUC
auc = metrics.roc_auc_score(y_train,pred_prob_train[:,1])
print('AUC for the Training Data: %.3f' % auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_train,pred_prob_train[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label = 'RF')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Training Data')


# AUC and ROC for the test data

# calculate AUC
test_auc = metrics.roc_auc_score(y_test,pred_prob_test[:,1])
print('AUC for the Test Data: %.3f' % test_auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_test,pred_prob_test[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label= 'No skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label='RF')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Test data')

# save and show the plot
plt.savefig('plots/RF_ROC.png')
#plt.show()


# ### Model 9: Bagging Classifier

# In[ ]:


# invoke the LogisticRegression function and find the bestfit model on training data
BGC_model = BaggingClassifier()
BGC_model.fit(X_train, y_train)
BGC_model.get_params()


# In[ ]:


## Performance Matrix on train data set
ytrain_predict = BGC_model.predict(X_train)
model_score = BGC_model.score(X_train, y_train)
print('\n \033[1m Train Accuracy Score : \033[0m', model_score)
print('\n \033[1m Train Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_train, ytrain_predict))
print('\n \033[1m Train Classification Report : \033[0m \n', metrics.classification_report(y_train, ytrain_predict))


# In[ ]:


## Performance Matrix on test data set
ytest_predict = BGC_model.predict(X_test)
test_model_score = BGC_model.score(X_test, y_test)
print('\n \033[1m Test Accuracy Score : \033[0m', test_model_score)
print('\n \033[1m Test Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_test, ytest_predict))
print('\n \033[1m Test Classification Report : \033[0m \n', metrics.classification_report(y_test, ytest_predict))


# In[ ]:


f,a =  plt.subplots(1,2,sharex=True,sharey=True,squeeze=False, figsize=(10, 5))

#Plotting confusion matrix for the Training Data and Test data

plot_0 = sns.heatmap((metrics.confusion_matrix(y_train, ytrain_predict)),annot=True,fmt='.5g',cmap='GnBu',ax=a[0][0]);
a[0][0].set_title('Training Data');

plot_1 = sns.heatmap((metrics.confusion_matrix(y_test, ytest_predict)),annot=True,fmt='.5g',cmap='OrRd',ax=a[0][1]);
a[0][1].set_title('Test Data');

plt.suptitle('Bagging Classifier Model');

plt.savefig('plots/BGC_CM.png')


# In[ ]:


# AUC and ROC for the training and testing data
# Training Data Probability Prediction
pred_prob_train = BGC_model.predict_proba(X_train)

# Test Data Probability Prediction
pred_prob_test = BGC_model.predict_proba(X_test)

# AUC and ROC for the train data

# calculate AUC
auc = metrics.roc_auc_score(y_train,pred_prob_train[:,1])
print('AUC for the Training Data: %.3f' % auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_train,pred_prob_train[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label = 'Bagged')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Training Data')


# AUC and ROC for the test data

# calculate AUC
test_auc = metrics.roc_auc_score(y_test,pred_prob_test[:,1])
print('AUC for the Test Data: %.3f' % test_auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_test,pred_prob_test[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label= 'No skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label='Bagged')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Test data')

# save and show the plot
plt.savefig('plots/BGC_ROC.png')
#plt.show()


# ### Model 10: Gradient Boosting Classifier

# In[ ]:


# invoke the LogisticRegression function and find the bestfit model on training data
GBC_model = GradientBoostingClassifier()
GBC_model.fit(X_train, y_train)
GBC_model.get_params()


# In[ ]:


## Performance Matrix on train data set
ytrain_predict = GBC_model.predict(X_train)
model_score = GBC_model.score(X_train, y_train)
print('\n \033[1m Train Accuracy Score : \033[0m', model_score)
print('\n \033[1m Train Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_train, ytrain_predict))
print('\n \033[1m Train Classification Report : \033[0m \n', metrics.classification_report(y_train, ytrain_predict))


# In[ ]:


## Performance Matrix on test data set
ytest_predict = GBC_model.predict(X_test)
test_model_score = GBC_model.score(X_test, y_test)
print('\n \033[1m Test Accuracy Score : \033[0m', test_model_score)
print('\n \033[1m Test Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_test, ytest_predict))
print('\n \033[1m Test Classification Report : \033[0m \n', metrics.classification_report(y_test, ytest_predict))


# In[ ]:


f,a =  plt.subplots(1,2,sharex=True,sharey=True,squeeze=False, figsize=(10, 5))

#Plotting confusion matrix for the Training Data and Test data

plot_0 = sns.heatmap((metrics.confusion_matrix(y_train, ytrain_predict)),annot=True,fmt='.5g',cmap='GnBu',ax=a[0][0]);
a[0][0].set_title('Training Data');

plot_1 = sns.heatmap((metrics.confusion_matrix(y_test, ytest_predict)),annot=True,fmt='.5g',cmap='OrRd',ax=a[0][1]);
a[0][1].set_title('Test Data');

plt.suptitle('Gradient Boosting Classifier');

plt.savefig('plots/GBC_CM.png')




# AUC and ROC for the training and testing data
# Training Data Probability Prediction
pred_prob_train = GBC_model.predict_proba(X_train)

# Test Data Probability Prediction
pred_prob_test = GBC_model.predict_proba(X_test)

# AUC and ROC for the train data

# calculate AUC
auc = metrics.roc_auc_score(y_train,pred_prob_train[:,1])
print('AUC for the Training Data: %.3f' % auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_train,pred_prob_train[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label = 'Boosted')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Training Data')


# AUC and ROC for the test data

# calculate AUC
test_auc = metrics.roc_auc_score(y_test,pred_prob_test[:,1])
print('AUC for the Test Data: %.3f' % test_auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_test,pred_prob_test[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label= 'No skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label='Boosted')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Test data')

# save and show the plot
plt.savefig('plots/GBC_ROC.png')
#plt.show()


# ## Model Tuning


# Tuned LR Model
dual=[True,False]
max_iter=[100,150,200]
solver = ['newton-cg', 'lbfgs', 'liblinear']
tol = [0.1, 0.01, 0.001]
param_grid = dict(dual=dual,
                  max_iter=max_iter, 
                  solver=solver, 
                  tol=tol)

import time

lr = LogisticRegression(penalty='l2')
LR_tuned_grid = GridSearchCV(estimator=lr, param_grid=param_grid, cv = 5, n_jobs=-1)

start_time = time.time()
grid_result = LR_tuned_grid.fit(X_train, y_train)
# Summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
print("Execution time: " + str((time.time() - start_time)) + ' ms')





## Performance Metrics on train data set
y_train_predict = LR_tuned_grid.predict(X_train)
model_score = LR_tuned_grid.score(X_train, y_train)
print('\n \033[1m Train Accuracy Score : \033[0m', model_score)
print('\n \033[1m Train Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_train, y_train_predict))
print('\n \033[1m Train Classification Report : \033[0m \n', metrics.classification_report(y_train, y_train_predict))





## Performance Metrics on test data set
y_test_predict = LR_tuned_grid.predict(X_test)
test_model_score = LR_tuned_grid.score(X_test, y_test)
print('\n \033[1m Test Accuracy Score : \033[0m', test_model_score)
print('\n \033[1m Test Confusion Matrix :  \033[0m \n', metrics.confusion_matrix(y_test, y_test_predict))
print('\n \033[1m Test Classification Report : \033[0m \n', metrics.classification_report(y_test, y_test_predict))




f,a =  plt.subplots(1,2,sharex=True,sharey=True,squeeze=False, figsize=(10, 5))

#Plotting confusion matrix for the Training Data and Test data

plot_0 = sns.heatmap((metrics.confusion_matrix(y_train, ytrain_predict)),annot=True,fmt='.5g',cmap='GnBu',ax=a[0][0]);
a[0][0].set_title('Training Data');

plot_1 = sns.heatmap((metrics.confusion_matrix(y_test, ytest_predict)),annot=True,fmt='.5g',cmap='OrRd',ax=a[0][1]);
a[0][1].set_title('Test Data');

plt.suptitle('Tuned Logistic Regression');

plt.savefig('plots/LRT_CM.png')




# AUC and ROC for the training and testing data
# Training Data Probability Prediction
pred_prob_train = LR_tuned_grid.predict_proba(X_train)

# Test Data Probability Prediction
pred_prob_test = LR_tuned_grid.predict_proba(X_test)

# AUC and ROC for the train data

# calculate AUC
auc = metrics.roc_auc_score(y_train,pred_prob_train[:,1])
print('AUC for the Training Data: %.3f' % auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_train,pred_prob_train[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label = 'Logit-Tuned')
# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Training Data')


# AUC and ROC for the test data

# calculate AUC
test_auc = metrics.roc_auc_score(y_test,pred_prob_test[:,1])
print('AUC for the Test Data: %.3f' % test_auc)

#  calculate roc curve
fpr, tpr, thresholds = metrics.roc_curve(y_test,pred_prob_test[:,1])
plt.figure(figsize=(4,4))
plt.plot([0, 1], [0, 1], linestyle='--', label= 'No skill')
# plot the roc curve for the model
plt.plot(fpr, tpr, marker='.',label='Logit-Tuned')

# axis labels
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='best')
plt.title('Test data')

# save and show the plot
plt.savefig('plots/LRT_ROC.png')
# #plt.show()



param_test = {'min_samples_split':range(1000,2100,200), 
              'min_samples_leaf':range(30,71,10),
              'max_depth':range(5,16,2),
              'n_estimators':range(50,160,50),
              'learning_rate':[0.1,0.01,0.001]}

GBC_gsearch = GridSearchCV(estimator = GradientBoostingClassifier(max_features='sqrt', random_state=10), 
                            param_grid = param_test, scoring='f1',n_jobs=-1, cv=5)

GBC_gsearch.fit(X_train, y_train)

GBC_gsearch.grid_scores_, GBC_gsearch.best_params_, GBC_gsearch.best_score_


# ## Feature Importance


#Print Feature Importance:
feat_imp = pd.Series(GBC_model.feature_importances_, X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(12,4))
feat_imp.plot(kind='bar', title='Feature Importances')

plt.ylabel('Feature Importance Score')


# ### The End




