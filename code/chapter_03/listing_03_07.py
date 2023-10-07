import pandas as pd

cleaned_data = my_data.dropna(
    subset=['d(135Ba/136Ba)', 'd(138Ba/136Ba)'])

print("Before cleaning: {} cols".format(my_data.shape[0]))
print("After cleaning: {} cols".format(cleaned_data.shape[0]))

''' 
Output:
Before cleaning: 19978 cols
After cleaning: 206 cols
'''