import os

import pandas as pd

for filename in os.listdir('elspotprices'):
    if filename.endswith("eur.csv"):
        filenm ='elspotprices/'+filename
        # filenm = 'elspotprices/'+ filename.split('.')[0]+'.xls'
        # os.rename(filenmcsv, filenm)
        data = pd.read_csv(filenm, error_bad_lines=False)
        print(data.head())
