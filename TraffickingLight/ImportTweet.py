import os
import json
import csv
import pandas as pd


#data_json = open('/Users/Ju1y/Documents/Trafficking Light/(Pimpin) OR (Pregofetish) OR (dfe)/tweets_1548882430_1.json', mode='r').read();
#data_python = json.loads(data_json)

path_to_json = '/Users/Ju1y/Documents/Trafficking Light/S11'
json_files = [pos_json for pos_json in os.listdir(path_to_json) if pos_json.endswith('.json')]

#change file num everytime by 1
filenum = 18
write_path = '/Users/Ju1y/Documents/Trafficking Light/S11/Training Data ' + str(filenum) + '.csv'


csv_out = open(write_path, mode='w') #open csv
writer = csv.writer(csv_out);

fields = ['time stamp', 'user', 'full text', 'image url' ]
writer.writerow(fields);

for pos_json in os.listdir(path_to_json):
    if pos_json.endswith('.json'):
        data_json = open(path_to_json+'/' + pos_json, mode='r').read()
        data_python = json.loads(data_json)
        for line in data_python:
            if 'extended_entities' in line:
                writer.writerow([line.get('created_at'),
                                 line.get('user').get('screen_name'),
                                 line.get('full_text').encode('unicode_escape'),
                                 line.get('extended_entities').get('media')[0].get('media_url')])
            else:
                writer.writerow([line.get('created_at'),
                                 line.get('user').get('screen_name'),
                                 line.get('full_text').encode('unicode_escape'),
                                 'no url'])

csv_out.close();

