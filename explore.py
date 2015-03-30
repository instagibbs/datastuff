import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import json
import pprint
import re
import copy
import nltk
from nltk.corpus import stopwords

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

#########################################################

#Printed schema to file for easier reading
f=open('schema_printed.txt', 'w')
pprint.pprint(data_schema, stream=f)
f=open('data_sample.txt', 'w')
pprint.pprint(data[0], stream=f)

##############################################################

'''
deadbills = 0
enact_dates = []
#Timescale of bills
for bill in data:
  yr =  time.strptime("2014-06-30", "%Y-%m-%d")#create time from fields
  print bill["actions"]["enacted"]
'''
#Ideas:
#******************************************************************************************************
# 1) Track changes in text over revisions, create distance metric, do regression/analysis on how it affects passage. bigrams,unigrams, etc. The thought is that bills that change a lot would have a 
#different chance of passage.
histCounter = [0]*5 #No bills had 5+ versions 
histSuccessCounter = [0]*5
for bill in data:
  histCounter[len(bill["documents"])] = histCounter[len(bill["documents"])] + 1
  if bill["actions"]["enacted"] != 'null':
    histSuccessCounter[len(bill["documents"])] = histSuccessCounter[len(bill["documents"])] + 1
# [0, 12667, 1400, 529, 173] total bills written
# [0, 3964, 0, 519, 171] successful bill passages per number of revisions

#Bills with revisions apparently only pass with 2+ revisions, and then almost always do so.
#The proposed task doesn't make sense. Since I still want to do something with text of bills
#I'll pivot on this new information and build a basic linear model to predict passage of 
#final versions of bills. I could also use predictor to see if first version of bills are 
#still deemed "positive" samples. 

#Basic cleaning of text to remove numbers, parens, etc.
stripre = "[\(\d\). -]"
stops = set(stopwords.words('english'))

complete_vocab = {}
for bill in data:
  words = re.split(stripre, bill['documents'][str(len(bill["documents"])-1)])
  for word in words:
    if word != "":
      complete_vocab[word.lower()] = complete_vocab.get(word.lower(), 0)+1
    
mincount = 20 #Word should occur a few times to count, lower dimensionality of problem
for vocab, count in complete_vocab.items():
  if count < mincount:
    del complete_vocab[vocab]

samples = []
binnedSamples = [ [] for i in range(5)]
for bill in data:
  sample = []#Structure will be binary word existance label at end
  sample_vocab = copy.deepcopy(complete_vocab)

  words = re.split(stripre, bill['documents'][str(len(bill["documents"])-1)])
  for word in words:
    word = word.lower()
    if word in sample_vocab:
      sample_vocab[word] = sample_vocab.get(word, 0)+1

  for k in complete_vocab.keys(): #Iterate using untouched dict for det ordering
    sample.append(sample_vocab[k])

  if bill["actions"]["enacted"] =='null':
    sample.append(0)
  else:
    sample.append(1)

  samples.append(sample)
  binnedSamples[len(bill["documents"])].append(sample)
  


#*****************************************************************************************************
# 2) Look at "bi-partisanship" vs "partisanship" on passage/voting metrics such as length of time from introduction to signing, etc
#Votes may be missing, so must check top-line details and trust that.

# 3) Look at when things get vetoed? Differential between passing lower/upper and vice versa to signing?



