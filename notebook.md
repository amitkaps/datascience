
[home](/MLcloud) | [amitkaps.com](http://amitkaps.com) | [bargava.com](http://bargava.com)


# Build & Deploy ML Models in Cloud

** Motivation for the Session**

- Solve a business problem
- Understand the end-to-end approach
- Build a data-driven Machine Learning application on the cloud

##### For code, go to [https://github.com/amitkaps/MLcloud](https://github.com/amitkaps/MLcloud)

** Our approach ** is to take a case-driven example to showcase this. And we will aim to go-wide vs. go-deep to do so. The approach will be both practical and scalable.


<br>

## INTRO


Lets start by understanding the overall approach for doing so.


```
                FRAME  ——> ACQUIRE  ——> REFINE ——>  
                                                  \
                                                TRANSFORM <——
                                                    ↑          ↘  
                                                    |        EXPLORE
                                                    ↓          ↗
                                                  MODEL   <——
                                                  /      
                BUILD <—— DEPLOY <—— INSIGHT <—— 

```



- **FRAME**: Problem definition
- **ACQUIRE**: Data ingestion 
- **REFINE**: Data wrangling
- **TRANSFORM**: Feature creation 
- **EXPLORE**: Feature selection 
- **MODEL**: Model creation
- **INSIGHT**: Model selection
- **DEPLOY**: Model deployment
- **BUILD**: Application building

<br>

## FRAME 

> "Doing data science requires more time thinking than doing."

A start-up providing loans to the consumer and has been running for the last few years. It is now planning to adopt a data-driven lens to its loan portfolio. What are the **type of questions** it can ask?
- What is the trend of loan defaults?
- Do older customers have more loan defaults?
- Which customer is likely to have a loan default?
- Why do customers default on their loan?


### Type of data-driven analytics
- **Descriptive**: Understand patterns, trends, deviations and outlier
- **Inquisitive**: Conduct hypothesis testing
- **Predictive**: Make a prediction
- **Causal**: Establish a causal link

**Our Question: What is the probability of a loan default?**

<br>

## ACQUIRE

> "Data is the new oil"

**Ways to acquire data** (typical data source)

- Download from an internal system
- Obtained from client, or other 3rd party
- Extracted from a web-based API
- Scraped from a website
- Extracted from a PDF file
- Gathered manually and recorded

**Data Formats**: flat files (e.g. csv, tsv, xls), databases (e.g. MySQL), streaming (e.g. json), storage (e.g. HDFS)


```python
#Load the libraries and configuration
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
```


```python
df = pd.read_csv("loan.csv") 
df.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>default</th>
      <th>amount</th>
      <th>grade</th>
      <th>years</th>
      <th>ownership</th>
      <th>income</th>
      <th>age</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1000</td>
      <td>B</td>
      <td>2.0</td>
      <td>RENT</td>
      <td>19200.0</td>
      <td>24</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>6500</td>
      <td>A</td>
      <td>2.0</td>
      <td>MORTGAGE</td>
      <td>66000.0</td>
      <td>28</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>2400</td>
      <td>A</td>
      <td>2.0</td>
      <td>RENT</td>
      <td>60000.0</td>
      <td>36</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>10000</td>
      <td>C</td>
      <td>3.0</td>
      <td>RENT</td>
      <td>62000.0</td>
      <td>24</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>4000</td>
      <td>C</td>
      <td>2.0</td>
      <td>RENT</td>
      <td>20000.0</td>
      <td>28</td>
    </tr>
  </tbody>
</table>
</div>



*Target*
- **default**: whether the applicant defaulted (1) or not (0)?

*Features* - Application Attributes
- **age**: age of the applicant
- **income**: annual income of the applicant
- **year**: no. of years of employment
- **ownership**: type of house owned
- **amount** : amount of loan requested by the applicant

*Features* - Behavioural Attributes:
- **grade**: credit grade of the applicant

## REFINE

> "Data is messy"

- **Remove** e.g. remove redundant data from the data frame
- **Derive** e.g. state and city from the location field
- **Parse** e.g. extract date from year and month column

Also, you need to check for consistency and quality of the data
- **Missing** e.g. Check for missing or incomplete data
- **Quality** e.g. Check for duplicates, accuracy, unusual data

Lets check for missing values in our data


```python
df.isnull().sum()
```




    default        0
    amount         0
    grade          0
    years        279
    ownership      0
    income         0
    age            0
    dtype: int64



Handling missing values


```python
df.dropna(axis=0, inplace=True) 
```

## TRANSFORM

> "What is measured may not help answer what is needed"

- **Convert** e.g. free text to coded value
- **Calculate** e.g. percentages, proportion
- **Merge** e.g. first and surname for full name
- **Aggregate** e.g. rollup by year, cluster by area
- **Filter** e.g. exclude based on location
- **Sample** e.g. extract a representative data
- **Summary** e.g. show summary stats like mean
- **Encoding** e.g. convert categorical variable - label, one-hot


```python
df['default'] = df['default'].astype('category')
```


```python
df['log_age'] = np.log(df.age)
df['log_income'] = np.log(df.income)
```

## EXPLORE


```python
from plotnine import *
%matplotlib inline
from plotnine.themes import theme_538
plot = ggplot(df) + theme_538()
```


```python
plot + aes('grade', fill ="default") + geom_bar(position = 'fill')
```


![png](notebook_files/notebook_17_0.png)





    <ggplot: (296598322)>




```python
plot + aes('grade', 'ownership', fill ="default") + geom_jitter(alpha = 0.2)
```


![png](notebook_files/notebook_18_0.png)





    <ggplot: (-9223372036555466748)>




```python
(
    ggplot(df) + 
    aes('grade', 'income', color = 'default') + 
    geom_jitter(alpha = 0.2) + geom_boxplot() +
    scale_y_log10() +
    facet_wrap('default')
)
```


![png](notebook_files/notebook_19_0.png)





    <ggplot: (-9223372036561380359)>




```python
plot + aes('log_age', 'log_income') + geom_bin2d() + facet_wrap('default')
```


![png](notebook_files/notebook_20_0.png)





    <ggplot: (301272310)>



### ** Model - Build a tree classifier **


```python
from sklearn import tree
from sklearn.externals import joblib
from firefly.client import Client
```


```python
X = df.loc[:,('age', 'income')]
y = df.loc[:,'default']
clf = tree.DecisionTreeClassifier(max_depth=10).fit(X,y)
joblib.dump(clf, "clf.pkl")
```




    ['clf.pkl']



### ** Build - the ML API **


```python
%%file simple.py
import numpy as np
from sklearn.externals import joblib
model = joblib.load("clf.pkl")

def predict(age, amount):
    features = [age, amount]
    prob0, prob1 = model.predict_proba([features])[0]
    return prob1
```

    Writing simple.py


### ** Deploy - the ML API **

Run the following command in your terminal 

     cd credit-risk/notebooks/
     firefly simple.predict

### ** Interact - get prediction using API**


```python
simple = Client("http://127.0.0.1:8000")
simple.predict(age=28, amount=10000)
```




    0.5373423860329777




```python
simple.predict(age=50, amount=240000)
```




    1.0




```python

```
