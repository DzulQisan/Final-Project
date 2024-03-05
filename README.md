# ✨ New York House : Predictive Analysis ✨ 
The data is related to direct marketing campaigns (phone calls) of a Portuguese banking institution. The classification goal is to predict if the client will subscribe to a term deposit. Link for the dataset : https://www.kaggle.com/henriqueyamahata/bank-marketing?select=bank-additional-names.txt . Thank God, this project was awarded as the "Best Final Project" by dibimbing.id, my data science bootcamp institution.

![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Microsoft PowerPoint](https://img.shields.io/badge/Microsoft_PowerPoint-B7472A?style=for-the-badge&logo=microsoft-powerpoint&logoColor=white)


Just in case you are not aware, the .ipynb file contains both the codes and the explanation, which you can also see easily [here](https://indrayantom.github.io/Bank_Marketing_Predictive/). Since this work was my final project in a Data Science Bootcamp, I also provide a Google Slide Presentation to summarize all the procedures and findings, see it [here](https://docs.google.com/presentation/d/1XxfgQliJreu22A_ZNEC0bhTklE0TXCUiNk4umcbErE0/edit?usp=sharing).

## Objectives 
Nowadays, banks can make money with a lot of different ways, even though at the core they are still considered as lenders. Generally, they make money by borrowing it from the depositors, who are compensated later with a certain interest rate and security for their funds. Then, the borrowed money will be lent out to borrowers who need it at the moment. The borrowers however, are charged with higher interest rate than what is paid to the depositors. The difference between interest rate paid and interest rate received is often called as interest rate spead, where the banks gain profit from.

The main object of this research is a certain Portuguese bank institution who was trying to collect money from the depositors through direct marketing campaigns. In general, direct marketing campaign requires in-house or out-sourced call centres. Even though the information of sales cost are not provided, several articles said that it could put a considerable strain on the expense ratio of the product. In this case, the sales team of the bank contacted about 40000 customers randomly, while only 11% (around 4500) of them were willing to deposit their money.

By assuming one direct marketing call for one customer costs the company 2 dollars and profits them 50  dollars. It can be said that from 2000 customers contacted, the bank will gain profit of 7000 dollars in total, knowing that 11 in every 100 random calls result in successful sales. However, the bank soughted for some approaches that will help them conduct more effective marketing campaign with better conversion rate, and machine learning is one of the answers.

**Business Objective**

Providing more detail to previous statement, this research is carried out to identify customers who are more likely to subscribe and build a machine learning model that is be able to predict the probability of a certain customer become a depositor, in order to help the sales team conducting more effective marketing campaign or higher profit. Keep in mind that based on the previous example, 10% increase in customer rate will be followed by almost 16% increase in profit. Conversion rate is used as the key metrics related to the evaluation of machine learning model.

## Libraries
Libraries such as [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [matplotlib](https://matplotlib.org/), and [seaborn](https://seaborn.pydata.org/) are the most commonly used in the analysis. However, I also used [sklearn](https://scikit-learn.org/stable/) to conduct the predictive analysis with some classification models.
```python
# ---- EDA + INSIGHT ----
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---- DATA PREP + REGRESSION ----
from statsmodels.stats.outliers_influence import variance_inflation_factor as vif
from statsmodels.tools.tools import add_constant
import statsmodels.api as sm
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression

# ---- DATA PREP + CLASSIFICATION ----
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

```

## Result Preview
Berdasarkan gambar yang Anda berikan, model multi-family home ini memiliki beberapa fitur penting:
![shapp_project](https://user-images.githubusercontent.com/92590596/156873400-f4e34d86-77e9-4461-b92d-70f811222be3.png)

**Fitur Penting:**

Lokasi: Terjangkau terletak di Richmond County, New York.
Tipe: Model ini adalah Multi-Family House.
Ukuran: Model ini memiliki luas rata-rata 2.228 kaki persegi.
Kamar Tidur: Model ini memiliki 4 kamar tidur.
Kamar Mandi: Model ini memiliki 3.5 kamar mandi.
**Kelebihan:**

Type ini memiliki banyak ruang untuk keluarga besar.
Type ini memiliki luas lahan.
**Kekurangan:**

Type ini memiliki harga yang relatif tinggi.
**Kesimpulan:**

Type multi-family home ini adalah pilihan yang baik bagi keluarga besar yang mencari rumah yang modern, fungsional, dan terletak di lokasi yang strategis. Namun, perlu diingat bahwa type ini memiliki harga yang relatif tinggi.

