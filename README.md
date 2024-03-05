# ✨ New York House : Analisis Dampak Jumlah Kamar Tidur dan Kamar Mandi terhadap Harga Rumah ✨ 
Analisis Dampak Jumlah Kamar Tidur dan Kamar Mandi terhadap Harga Rumah.

https://www.kaggle.com/datasets/nelgiriyewithana/new-york-housing-market

Kumpulan data ini berisi harga rumah di New York, yang memberikan wawasan berharga mengenai pasar real estate di wilayah tersebut. Ini mencakup informasi seperti nama broker, tipe rumah, harga, jumlah kamar tidur dan kamar mandi, luas properti, alamat, negara bagian, wilayah administratif dan lokal, nama jalan, dan koordinat geografis.


*   BROKERTITLE: Title of the broker
*   TYPE: Type of the house
*   PRICE: Price of the house
*   BEDS: Number of bedrooms
*   BATH: Number of bathrooms
*   PROPERTYSQFT: Square footage of the property
*   ADDRESS: Full address of the house
*   STATE: State of the house
*   MAIN_ADDRESS: Main address information
*   ADMINISTRATIVE_AREA_LEVEL_2: Administrative area level 2 information
*   LOCALITY: Locality information
*   SUBLOCALITY: Sublocality information
*   STREET_NAME: Street name
*   LONG_NAME: Long name
*   FORMATTED_ADDRESS: Formatted address
*   LATITUDE: Latitude coordinate of the house
*   LONGITUDE: Longitude coordinate of the house
  
![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Microsoft PowerPoint](https://img.shields.io/badge/Microsoft_PowerPoint-B7472A?style=for-the-badge&logo=microsoft-powerpoint&logoColor=white)


Just in case you are not aware, the .ipynb file contains both the codes and the explanation, which you can also see easily [here](https://github.com/DzulQisan/Final-Project/blob/master/code/Finpro_Dzulqisan_Maulana%20ML.ipynb). Since this work was my final project in a Data Science Bootcamp, I also provide a Google Slide Presentation to summarize all the procedures and findings, see it [here](https://github.com/DzulQisan/Final-Project/blob/master/Finpro_Dzulqisan%20Maulana.pptx).

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
![Decision Tree](https://github.com/DzulQisan/Final-Project/blob/master/assets/D_Tree%20Feature%20target%20Type%20Multy%20Family%20Home.jpg)

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

