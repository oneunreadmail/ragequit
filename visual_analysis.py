import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns

pd.set_option('display.width', 256)

x = np.random.normal(size=100)
sns.distplot(x);