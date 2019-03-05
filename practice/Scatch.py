
import json
import csv


data_json = open('/Users/Ju1y/git/TwitterAPI/TwitterAPI/(gfe) OR (RHGO) OR (Bottom bitch) OR (Stompdown)/tweets_1545250754_1.json', mode='r').read();
data_python = json.loads(data_json);

csv_out = open('Training Data.csv', mode='w'); #open csv
writer = csv.writer(csv_out);

fields = ['time stamp', 'user', 'full text', 'image url' ]
writer.writerow(fields);

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

