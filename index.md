
[home](/MLcloud) | [amitkaps.com](http://amitkaps.com) | [bargava.com](http://bargava.com)


# Build & Deploy ML Models in Cloud
---

** Motivation for the Session**

- Solve a business problem
- Understand the end-to-end approach
- Build a data-driven Machine Learning application on the cloud

** Our approach ** is to take a case-driven example to showcase this. And we will aim to go-wide vs. go-deep to do so. The approach will be both practical and scalable.

<br>
<br>

## INTRO
---

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
<br>


## FRAME 
---

> "Doing data science requires quite a bit of thinking and we believe that when you’ve completed a good data science analysis, you’ve spent more time thinking than doing." - Roger Peng

A start-up providing loans to the consumer and has been running for the last few years. It is now planning to adopt a data-driven lens to its loan portfolio. What are the **type of questions** it can ask?
- What is the trend of loan defaults?
- Do older customers have more loan defaults?
- Which customer is likely to have a loan default?
- Why do customers default on their loan?


####  Type of data-driven analytics
- **Descriptive**: Understand patterns, trends, deviations and outlier
- **Inquisitive**: Conduct hypothesis testing
- **Predictive**: Make a prediction
- **Causal**: Establish a causal link

**Our Question: What is the probability of a loan default?**

<br>

<br>

## Acquire
---

> "Data is the new oil"

**Ways to acquire data** (typical data source)

- Download from an internal system
- Obtained from client, or other 3rd party
- Extracted from a web-based API
- Scraped from a website
- Extracted from a PDF file
- Gathered manually and recorded

**Data Formats**
- Flat files (e.g. csv)
- Excel files
- Database (e.g. MySQL)
- JSON
- HDFS (Hadoop)


```
#Load the libraries and configuration
import numpy as np
import pandas as pd
```


```
df = pd.read_csv("loan.csv") 
```

### **Refine - drop NAs**



```
df.dropna(axis=0, inplace=True) 
```

### ** Transform - log scale **


```
df['log_age'] = np.log(df.age)
df['log_income'] = np.log(df.income)
```

### ** Explore - age, income & default **


```
from plotnine import *
%matplotlib inline
```

    /Users/amitkaps/miniconda3/lib/python3.6/site-packages/statsmodels/compat/pandas.py:56: FutureWarning: The pandas.core.datetools module is deprecated and will be removed in a future version. Please use the pandas.tseries module instead.
      from pandas.core import datetools



```
plt.matshow(df.corr())
```


    -----------------------------------------------------------------------

    NameError                             Traceback (most recent call last)

    <ipython-input-70-61b8406b1e2d> in <module>()
    ----> 1 plt.matshow(df.corr())
    

    NameError: name 'plt' is not defined



```
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
      <th>log_age</th>
      <th>log_income</th>
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
      <td>3.178054</td>
      <td>9.862666</td>
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
      <td>3.332205</td>
      <td>11.097410</td>
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
      <td>3.583519</td>
      <td>11.002100</td>
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
      <td>3.178054</td>
      <td>11.034890</td>
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
      <td>3.332205</td>
      <td>9.903488</td>
    </tr>
  </tbody>
</table>
</div>




```
df['default'] = df['default'].astype('category')

```


```
ggplot(df) + aes('grade', fill ="default") + geom_bar(position = 'fill')
```


![png](index_files/index_16_0.png)





    <ggplot: (292944062)>




```
ggplot(df) + aes('grade', 'ownership', fill ="default") + geom_jitter(alpha = 0.2)
```


![png](index_files/index_17_0.png)





    <ggplot: (-9223372036561379342)>




```
ggplot(df) + aes('ownership', fill ="default") + geom_bar(position = 'fill')
```


![png](index_files/index_18_0.png)





    <ggplot: (-9223372036554328979)>




```
(
  ggplot(df) + 
  aes('years', '..count..', color = 'default') + 
  geom_freqpoly()
)
```

    /Users/amitkaps/miniconda3/lib/python3.6/site-packages/plotnine/stats/stat_bin.py:90: UserWarning: 'stat_bin()' using 'bins = 101'. Pick better value with 'binwidth'.
      warn(msg.format(params['bins']))



![png](index_files/index_19_1.png)





    <ggplot: (293366820)>




```
(
  ggplot(df) + 
  aes('amount', '..count..', color = 'default') + 
  geom_freqpoly(binwidth = 0.05) +
  scale_x_log10()
)
```


![png](index_files/index_20_0.png)





    <ggplot: (299466111)>




```
(
  ggplot(df) + 
  aes('income', '..count..', color = 'default') + 
  geom_freqpoly(binwidth = 0.05) +
  scale_x_log10()
)
```


![png](index_files/index_21_0.png)





    <ggplot: (301039624)>




```
(
    ggplot(df) + 
    aes('grade', 'income', color = 'default') + 
    geom_jitter(alpha = 0.2) + geom_boxplot() +
    scale_y_log10() +
    facet_wrap('default')
)
```


![png](index_files/index_22_0.png)





    <ggplot: (-9223372036555608828)>




```
ggplot(df) + aes('grade') + geom_histogram()
```

    /Users/amitkaps/miniconda3/lib/python3.6/site-packages/plotnine/stats/stat_bin.py:90: UserWarning: 'stat_bin()' using 'bins = 30'. Pick better value with 'binwidth'.
      warn(msg.format(params['bins']))



![png](index_files/index_23_1.png)





    <ggplot: (303137800)>




```
ggplot(df) + aes('log_age', 'log_income') + geom_bin2d() + facet_wrap('default')
```


![png](index_files/index_24_0.png)





    <ggplot: (299031395)>



### ** Model - Build a tree classifier **


```
from sklearn import tree
from sklearn.externals import joblib
from firefly.client import Client
```


```
X = df.loc[:,('age', 'income')]
y = df.loc[:,'default']
clf = tree.DecisionTreeClassifier(max_depth=10).fit(X,y)
joblib.dump(clf, "clf.pkl")
```




    ['clf.pkl']



### ** Build - the ML API **


```
%%file simple.py
import numpy as np
from sklearn.externals import joblib
model = joblib.load("clf.pkl")

def predict(age, amount):
    features = [age, amount]
    prob0, prob1 = model.predict_proba([features])[0]
    return prob1
```

    Overwriting simple.py


### ** Deploy - the ML API **

Run the following command in your terminal 

     cd credit-risk/notebooks/
     firefly simple.predict

### ** Interact - get prediction using API**


```
simple = Client("http://127.0.0.1:8000")
simple.predict(age=28, amount=10000)
```




    0.5373423860329777




```
simple.predict(age=50, amount=240000)
```




    1.0




```

```
