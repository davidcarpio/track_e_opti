import pandas as pd
import numpy as np

file_path = 'data/Power Unit/Carto pricess III.xlsx'
df = pd.read_excel(file_path, sheet_name='Feuil1')
print(df.iloc[30:50, 0:10].to_string())
