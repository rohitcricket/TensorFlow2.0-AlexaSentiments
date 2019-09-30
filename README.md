# Classify Alexa User Sentiments Using TensorFlow 2.0

![Echo Dot](echo-dot.jpg)

This project classifies user sentiment of Alexa devices using ANN (Artificial Neural Networks) and [Tensorflow](https://www.tensorflow.org) 2.0. 

### Data Reference:

Dataset consists of 3000 Amazon customer reviews, star ratings, date of review, variant and feedback of various amazon Alexa products like Alexa Echo, Echo dots.

The objective is to discover insights into consumer reviews and perfrom sentiment analysis on the data.
Dataset: www.kaggle.com/sid321axn/amazon-alexa-reviews

You can find the data in this directory. The file is: amazon-alexa.tsv

### Step 1: Open a [Colab](https://colab.research.google.com) python notebook

### Step 2: Import TensorFlow and Python Libraries


```
!pip install tensorflow-gpu==2.0.0.alpha0
import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

### Step 3: Import the dataset

You will need to mount your drive using the following commands:
For more information regarding mounting, please check this out [here](https://stackoverflow.com/questions/46986398/import-data-into-google-colaboratory).


```
from google.colab import drive
drive.mount('/content/drive')
```

Upload the data file from Kaggle to your Google drive and then access it

```
df_alexa = pd.read_csv('/content/drive/My Drive/Colab Notebooks/amazon_alexa.tsv', sep='\t')
```

Get more information about your dataset
```
df_alexa.info()
df_alexa.describe()
df_alexa.head(10)
df_alexa.tail(10)
```

### Step 4: Visualize the dataset using Seaborn, a python library
See more steps in the colab.

### Step 5: Create testing and training data set and clean the data. 
See steps in the colab.

### Step 6: Train the Model. 
See steps in the colab.

### Step 7: Evaluate the Model. 
See steps in the colab.

### Step 8: Improve the Model
If you are not satisfied with the results, then you can increase the number of independent variables and retrain the same model. See steps in the colab.