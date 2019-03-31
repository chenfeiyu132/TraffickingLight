
from sklearn.feature_extraction.text import TfidfVectorizer
import sys
import pandas as pd
import twitter


def flag_input(df):
    flags = []
    line = 164
    for row in range(total_rows): #create new array
        flags.append('N')
    for row in range(total_rows):
        try:
            print(line, (row+1), " ",  df.loc[row,'full text'])
            text = input('Y/M/N?(Yes, maybe, or no?)')
            line += 1
            flags[row] = text
        except KeyboardInterrupt: #if stopped midway, it would stop operation and break out the loop: have to restart next time.
            break
    return flags



csv_in = pd.read_csv('/Users/Ju1y/Documents/Trafficking Light/S8/Training Data 15.csv')
total_rows = len(csv_in['full text'])

csv_in['Flag'] = flag_input(csv_in)
csv_in.to_csv('/Users/Ju1y/Documents/Trafficking Light/Flagged/Training Data Flagged 10.csv', index = False) #writing back to new csv file





