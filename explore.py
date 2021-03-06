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
import collections
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from sklearn import svm
import numpy as np
import csv
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import GaussianNB
#from sklearn.ensemble import AdaBoostClassifier
from sklearn import neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFECV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import metrics
from sklearn.cross_validation import train_test_split

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
print "Data read in, samples written to disk"
##############################################################

'''
deadbills = 0
enact_dates = []
#Timescale of bills
for bill in data:
  yr =  time.strptime("2014-06-30", "%Y-%m-%d")#create time from fields
  print bill["actions"]["enacted"]
'''

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
'''
#Bills with revisions apparently only pass with 2+ revisions, and then almost always do so.
#The proposed task doesn't make sense. Since I still want to do something with text of bills
#I'll pivot on this new information and build a basic linear model to predict passage of 
#final versions of bills. I could also use predictor to see if first version of bills are 
#still deemed "positive" samples. 

#Basic cleaning of text to remove numbers, parens, etc.
stripre = "[\(\d\). -:_,;]"
stops = set(stopwords.words('english'))

complete_vocab = {}
bagsofwords = []
for bill in data:
  billtext = bill["documents"][str(len(bill["documents"])-1)].lower()
  bagsofwords.append(collections.Counter(re.findall(r'[a-zA-Z]+', billtext)))

for bag in bagsofwords:
  for word, count in bag.items():
    complete_vocab[word] = complete_vocab.get(word, 0) + count

print "Full dictionary completed."
    
'''
complete_vocab = {}
for bill in data:
  words = re.split(stripre, bill['documents'][str(len(bill["documents"])-1)])
  for word in words:
    if word != "":
      complete_vocab[word.lower()] = complete_vocab.get(word.lower(), 0)+1
'''
mincount = 50 #Word should occur a few times to count, lower dimensionality of problem
for vocab, count in complete_vocab.items():
  if count < mincount:
    del complete_vocab[vocab]
print "Dictionary pruned"
'''
sample_vocab_template = copy.deepcopy(complete_vocab)
for k in sample_vocab_template.keys():
  sample_vocab_template[k] = 0
'''
'''
samples=[]
labels=[]
#samples=np.zeros((len(bagsofwords), len(complete_vocab)))
#samples=csr_matrix((len(bagsofwords), len(complete_vocab)))
samples=lil_matrix((len(bagsofwords), len(complete_vocab)))
other_feats = []
for i, (bag, bill) in enumerate(zip(bagsofwords, data)):
  other_feats.append(bill["state"])
  sample_vector=[]
  for word in complete_vocab:
    sample_vector.append(bag[word])
  if bill["actions"]["enacted"] =='null':
    labels.append(0)
  else:
    labels.append(1)
  for k in range(len(complete_vocab)): #Doing it this way otherwise takes forever
    if sample_vector[k] != 0:
      samples[i,k] = sample_vector[k] 
  if i % 100 == 0:
    print i
'''  
other_feats=[]
for bill in data:
  other_feats.append(bill['state'])

md_data = []
ca_data = []
for feat in other_feats:
  if feat == 'md':
    md_data.append(1)
    ca_data.append(0)
  elif feat == 'ca':
    md_data.append(0)
    ca_data.append(1)
  else:
    md_data.append(0)
    ca_data.append(0)

labels = np.asarray(labels)
X_scaled = preprocessing.scale(samples, with_mean=False)
x_train, x_test, y_train, y_test = train_test_split(X_scaled, labels, test_size=0.33)

#*******************************
#MD/CA split experiments
'''
x_md_train, x_md_test, y_md_train, y_md_test = train_test_split(X_scaled[np.nonzero(md_data)[0],:], labels[np.nonzero(md_data)[0]], test_size=0.33)
x_ca_train, x_ca_test, y_ca_train, y_ca_test = train_test_split( X_scaled[np.nonzero(ca_data)[0],:], labels[np.nonzero(ca_data)[0]], test_size=0.33)

clf_l2_LR_md = LogisticRegression(C=.01, penalty='l2', tol=0.01)
clf_l2_LR_md.fit(x_md_train, y_md_train)

clf_l2_LR_ca = LogisticRegression(C=.01, penalty='l2', tol=0.01)
clf_l2_LR_ca.fit(x_ca_train, y_ca_train)

l2_md_ca_preds = clf_l2_LR_md.predict(x_ca_test)
l2_md_md_preds = clf_l2_LR_md.predict(x_md_test)
l2_ca_md_preds = clf_l2_LR_ca.predict(x_md_test)
l2_ca_ca_preds = clf_l2_LR_ca.predict(x_ca_test)

print classification_report(y_ca_test, l2_md_ca_preds)
print classification_report(y_md_test, l2_md_md_preds)
print classification_report(y_md_test, l2_ca_md_preds)
print classification_report(y_ca_test, l2_ca_ca_preds)

#*********************************

np.save("x_train_" + str(mincount)+ ".npy",x_train)
np.save("x_test_" + str(mincount)+ ".npy",x_test)
np.save("y_train_" + str(mincount)+ ".npy",y_train)
np.save("y_test_" + str(mincount)+ ".npy",y_test)

print "Logistic Regression"
clf_l1_LR = LogisticRegression(C=.01, penalty='l1', tol=0.01)
clf_l2_LR = LogisticRegression(C=.01, penalty='l2', tol=0.01)
clf_l1_LR.fit(x_train, y_train)
clf_l2_LR.fit(x_train, y_train)

l1_preds = clf_l1_LR.predict(x_test)
l2_preds = clf_l2_LR.predict(x_test)
print "L1",classification_report(y_test, l1_preds)
print "L2",classification_report(y_test, l2_preds)

l2_pred_proba = clf_l2_LR.predict_proba(x_test)[:,1]
l1_pred_proba = clf_l1_LR.predict_proba(x_test)[:,1]

tuned_parameters = [{'C': [.0001, .001, .01, 1, 10, 100]}]
best1 = GridSearchCV(LogisticRegression(penalty='l2', tol=0.01), tuned_parameters, cv=3)
best1.fit(x_train, y_train)
print best1.best_estimator_ #With 20 cutoff, C=.01 for l1 is best

  #samples.append(sample_vector)
'''
#*****************************************************************************************************
# 2) Look at "bi-partisanship" vs "partisanship" on passage/voting metrics such as length of time from introduction to signing, etc
#Votes may be missing, so must check top-line details and trust that.
partisan_fracs = []
rep_affiliation = {}
for bill in data:
  sponsors = bill["sponsors"]
  dem=0.00001 #eps to not div by 0
  rep=0.00001
  for sponsor in sponsors:
    #print sponsor
    if sponsor["party"] != None and sponsor["party"][0:10] == 'Republican':
      rep += 1
      rep_affiliation[sponsor['name']] = 'r'
    elif sponsor["party"] != None and sponsor["party"][0:10] == 'Democratic':
      dem += 1
      rep_affiliation[sponsor['name']] = 'd'
  partisan_frac = float(dem)/rep if rep>dem else float(rep)/dem
  partisan_fracs.append(1-partisan_frac) 

partisan_fracs = np.asarray(partisan_fracs)
plt.scatter(partisan_fracs, labels)
plt.savefig("scatter.png")
plt.clf()

md_labels=labels[np.nonzero(md_data)[0]]
ca_labels=labels[np.nonzero(ca_data)[0]]
md_partisan_fracs=partisan_fracs[np.nonzero(md_data)[0]]
ca_partisan_fracs=partisan_fracs[np.nonzero(ca_data)[0]]

plt.hist(partisan_fracs[np.nonzero(labels)[0]], bins = 100)
plt.savefig("pos.png")
plt.clf()
plt.hist(partisan_fracs[np.where(labels==0)[0]], bins = 100)
plt.savefig("neg.png")
plt.clf()

#Split states, see difference

plt.hist(md_partisan_fracs[np.nonzero(md_labels)[0]], bins = 100)
plt.savefig("pos_md.png")
plt.clf()
plt.hist(md_partisan_fracs[np.where(md_labels==0)[0]], bins = 100)
plt.savefig("neg_md.png")
plt.clf()

plt.hist(ca_partisan_fracs[np.nonzero(ca_labels)[0]], bins = 100)
plt.savefig("pos_ca.png")
plt.clf()
plt.hist(ca_partisan_fracs[np.where(ca_labels==0)[0]], bins = 100)
plt.savefig("neg_ca.png")
plt.clf()

#Look at where in distribution Pos/Neg diverges the most?

# 3) Clustering representatives from a similarity graph

#Construct similarity graph
md_reps = set([])
ca_reps = set([])


#Get sets of reps
for i, bill in enumerate(data):
  for vote_chamber in bill["votes"].keys():
    for yes in bill["votes"][vote_chamber]['yes_vote']:
      if md_data[i] == 1:
        md_reps.add(yes)
      elif ca_data[i] == 1:
        ca_reps.add(yes)
    for no in bill["votes"][vote_chamber]['no_vote']:
      if md_data[i] == 1:
        md_reps.add(no)
      elif ca_data[i] == 1:
        ca_reps.add(no)

#Construct similarity matrices for reps
md_rep_mat = np.zeros((len(md_reps), len(md_reps)))
md_dict = {}
for i, rep in enumerate(md_reps):
  md_dict[rep] = i

ca_rep_mat = np.zeros((len(ca_reps), len(ca_reps)))
ca_dict = {}
for i, rep in enumerate(ca_reps):
  ca_dict[rep] = i

#Count co-votes as strength of connection. Double-counting I believe.
for i, bill in enumerate(data):
  for vote_chamber in bill["votes"].keys():
    for yes in bill["votes"][vote_chamber]['yes_vote']:
      for yes2 in bill["votes"][vote_chamber]['yes_vote']:
        if yes != yes2:
          if md_data[i] == 1:
            md_rep_mat[md_dict[yes], md_dict[yes2]] += 1
          elif ca_data[i] == 1:
            ca_rep_mat[ca_dict[yes], ca_dict[yes2]] += 1
    for no in bill["votes"][vote_chamber]['yes_vote']:
      for no2 in bill["votes"][vote_chamber]['yes_vote']:
        if no != no2:
          if md_data[i] == 1:
            md_rep_mat[md_dict[no], md_dict[no2]] += 1
          elif ca_data[i] == 1:
            ca_rep_mat[ca_dict[no], ca_dict[no2]] += 1
        
spectral = SpectralClustering(n_clusters=2, affinity='precomputed')
spectral.fit(md_rep_mat) 
ys_md = spectral.fit_predict(md_rep_mat)

spectral = SpectralClustering(n_clusters=2, affinity='precomputed')
spectral.fit(ca_rep_mat) 
ys_ca = spectral.fit_predict(ca_rep_mat)

y_yhat_md = []
y_yhat_ca = []
#See if sponsors vote with people from their own party the most
for rep, party in rep_affiliation.items():
  entry = []
  if rep in md_dict:
    entry.append(party)
    entry.append(ys_md[md_dict[rep]])
    y_yhat_md.append(entry)
  elif rep in ca_dict:
    entry.append(party)
    entry.append(ys_ca[ca_dict[rep]])
    y_yhat_ca.append(entry)
#  else:
#    print rep, "not found"

d0count = 0
d1count = 0
for entry in y_yhat_md:
  if entry[0] == 'd' and entry[1] == 0 or entry[0] == 'r' and entry[1] == 1:
    d0count += 1
  else:
    d1count += 1
    
d0count = 0
d1count = 0
for entry in y_yhat_ca:
  if entry[0] == 'd' and entry[1] == 0 or entry[0] == 'r' and entry[1] == 1:
    d0count += 1
  else:
    d1count += 1
