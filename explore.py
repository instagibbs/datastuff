import numpy as np
import json
import pprint

#Python puking if I try to load the whole thing at once, load line by line instead
#json_data=open("data.json").read()
data = []
with open('data.json') as f:
  for line in f:
    data.append(json.loads(line))


#Read in schema
json_schema=open("data_schema.json").read()

#data = json.loads(json_data)
data_schema = json.loads(json_schema)

#Printed schema to file for easier reading
f=open('schema_printed.txt', 'w')
pprint.pprint(data_schema, stream=f)
f=open('data_sample.txt', 'w')
pprint.pprint(data[0], stream=f)

#Next, I want to take a look at a histogram of various counts of types of bills, etc.
#Time, type of bill, sessions of house/senate, etc

#Ideas:

#Track changes in text over revisions, create distance metric, do regression/analysis on how it affects passage. bigrams,unigrams, etc.

#Look at "bi-partisanship" vs "partisanship" on passage/voting metrics such as length of time from introduction to signing, etc
#Votes may be missing, so must check top-line details and trust that.

#Look at when things get vetoed? Differential between passing lower/upper and vice versa to signing?