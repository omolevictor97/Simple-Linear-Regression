#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

pumpkins = pd.read_csv("US-pumpkins.csv")

# we are selecting the pumpkins package that contains a string of "bushel" and making sure that the case is equal to True

pumpkins = pumpkins[pumpkins["Package"].str.contains("bushel", case=True, regex=True)]

pumpkins.head()


# In[2]:


pumpkins.isnull().sum()


# In[3]:


# since we would not be needing all of these columns we shall drop the columns we do not want

new_columns = ["Package", "Month", "Low Price", "High Price", "Date", "Variety", "City Name"]

pumpkins = pumpkins.drop([c for c in pumpkins.columns if c not in new_columns], axis=1)

# we shall also create price average variable to store

price = (pumpkins["Low Price"] + pumpkins["High Price"]) / 2

# convert months to their corresponding number

month = pd.DatetimeIndex(pumpkins["Date"]).month
day_of_year = pd.to_datetime(pumpkins["Date"]).apply(lambda dt: (dt - datetime(dt.year,1,1)).days)

#create a new pumpkn dataframe

new_pumpkins = pd.DataFrame({"Month": month, "Package": pumpkins["Package"], "Low Price": pumpkins["Low Price"],"High Price": pumpkins["High Price"], "Price": price, "DayOfYear": day_of_year, "Variety": pumpkins["Variety"], "City":                            pumpkins["City Name"]})

# we will also like to normalize the price per bushel weight

new_pumpkins.loc[new_pumpkins["Package"].str.contains("1 1/9"), "Price"] = price/(1/9)
new_pumpkins.loc[new_pumpkins["Package"].str.contains("1 1/9"), "Price"] = price/ (1/2)

new_pumpkins.head()


# In[4]:


import seaborn as sns

price = new_pumpkins["Price"]
month = new_pumpkins["Month"]

plt.scatter(price, month, color="red")


# In[5]:


low_price = new_pumpkins["Low Price"]
plt.scatter(price, low_price, color="black")

plt.show()


# In[6]:


columns = ["Month", "Package", "Low Price", "High Price"]

for c in columns:
    plt.figure(figsize=(9,6))
    col = new_pumpkins[c]
    price = new_pumpkins["Price"]
    
    plt.scatter(price, col, color="green")
    plt.xlabel("price")
    plt.ylabel(c)
    
    plt.title("Price against " +c )
    plt.show


# In[7]:


new_pumpkins.groupby(["Month"])["Price"].mean().plot(kind ="bar")


# In[8]:


ax = None

colors = ['red','blue','green','yellow']
for i, var in enumerate(new_pumpkins.Variety.unique()):
    df = new_pumpkins[new_pumpkins["Variety"] == var]
    ax = df.plot.scatter("DayOfYear", "Price", ax=ax, c=colors[i], label=var)


# In[9]:


for i in new_pumpkins.Variety.unique():
    df = new_pumpkins[new_pumpkins["Variety"] == i]
    
    df.plot.scatter("DayOfYear", "Price", color="red", label=i)


# In[10]:


for i in new_pumpkins["Variety"].unique():
    
    new_pumpkins[i] = new_pumpkins["Variety"] == i
    new_pumpkins[i].dropna(inplace=True)
    new_pumpkins[i].info()


# In[11]:


pie_pumpkins = new_pumpkins[new_pumpkins["Variety"] == "PIE TYPE"]


# In[12]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X = pie_pumpkins["DayOfYear"].to_numpy().reshape(-1,1)

y = pie_pumpkins["Price"]


# In[13]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# In[14]:


LR = LinearRegression().fit(X_train, y_train)


# In[15]:


y_pred = LR.predict(X_test)

print(y_pred)


# In[16]:


mse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(y_pred)*100:3.3}%)')


# In[17]:


score = LR.score(X_train, y_train)

print("Model determination {}".format(score))


# In[18]:


plt.scatter(X_test, y_test)
plt.plot(X_test, y_pred)


# <strong>Our line of fit doesn't estimate our model well, we shall love to use some other estimators to create a curve
# line of fit for our model</strong>

# In[19]:


from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline


# In[20]:


pipeline = make_pipeline(PolynomialFeatures(3), StandardScaler(), LinearRegression(), verbose=True)


# In[21]:


pipe_model = pipeline.fit(X_train, y_train)


# In[22]:


y_pred = pipe_model.predict(X_test)


# In[23]:


mse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(y_pred)*100:3.3}%)')


# In[24]:


score = pipe_model.score(X_train, y_train)

print("Model Determination {}".format(score))


# <strong>We might want to predict for all of the variety using one model and so we will be "One Hot Encoding" all of the categorical variables as our features, thus we use a pd.get_dummies on our categorical and leave the numerical features as they were</strong>

# In[44]:


X = pd.get_dummies(new_pumpkins["Variety"])         .join(new_pumpkins["Month"])        .join(pd.get_dummies(new_pumpkins["City"]))         .join(pd.get_dummies(new_pumpkins["Package"]))

y = new_pumpkins["Price"]


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 30)


# In[46]:


pipeline = make_pipeline(PolynomialFeatures(2), StandardScaler(), LinearRegression(), verbose=True)

pipeline_model = pipeline.fit(X_train, y_train)


# In[47]:


y_pred = pipeline_model.predict(X_test)

mse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Mean error: {mse:3.3} ({mse/np.mean(y_pred)*100:3.3}%)')


# In[48]:


score = pipeline_model.score(X_train, y_train)

print("The model determination is {:.2f}%".format(score * 100))


# In[ ]:




