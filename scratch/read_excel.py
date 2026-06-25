import pandas as pd
file_path = 'data/Power Unit/Carto pricess III.xlsx'
xls = pd.ExcelFile(file_path)
for sheet in xls.sheet_names:
    print(f"--- Sheet: {sheet} ---")
    df = pd.read_excel(xls, sheet_name=sheet)
    print(df.head(10).to_string())
