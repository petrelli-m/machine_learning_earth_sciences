import pandas as pd

my_data = pd.read_excel("PGD_SiC_2021-01-10.xlsx", sheet_name='PGD-SIC') 
print(my_data.info(memory_usage="deep"))

''' 
Output:
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 19978 entries, 0 to 19977
Columns: 123 entries, PGD ID to err[d(138Ba/136Ba)]
dtypes: float64(112), object(11)
memory usage: 29.4 MB
'''
