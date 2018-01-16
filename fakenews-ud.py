import re, csv, os, nltk, subprocess, urllib.request
import numpy as np
from operator import itemgetter
from matplotlib import pyplot as plt
from matplotlib import rcParams
from nltk import word_tokenize
from bs4 import BeautifulSoup
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime 
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
startTime= datetime.now() 


#######################
#Hack to generate CoNLL-U files with UD deps
#######################


def udSP(filename, t= None):
	statement="java -mx1g edu.stanford.nlp.parser.lexparser.LexicalizedParser -retainTMPSubcategories -outputFormat 'wordsAndTags,penn,typedDependencies' edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz " + filename
	statement2='java -mx1g  edu.stanford.nlp.parser.lexparser.LexicalizedParser -outputFormat "penn" edu/stanford/nlp/models/lexparser/englishPCFG.ser.gz ' + filename +' > '+ os.getcwd() +'/' + t + '.tree'
	print('#######################')
	print('Starting subprocesses')
	p1 = subprocess.Popen(statement, shell=True)
	p1.wait()
	p2 = subprocess.Popen(statement2, shell=True)
	p2.wait()
	p3 = subprocess.Popen('java -mx1g edu.stanford.nlp.trees.ud.UniversalDependenciesConverter -treeFile '+ os.getcwd() +'/'+ t+ '.tree > ' +t +'.conllu', shell=True)
	p3.wait()
	p3.kill()
	cfile= t + '.conllu'
	return cfile


#######################
#For reading fake/real data
#######################

def bf(path, t= None):
	print('#######################')
	print('Getting text')
	print('Path:', path)
	dic={}
	ds=[x for x in os.listdir(path) if x!='.DS_Store']
	sub_rf=[]
	for i in ds:
		p= path + '/' + i
		sub_rf.append(p)

	for p in sub_rf:
		files=os.listdir(p)
		if '.DS_Store' in files:
			files.remove('.DS_Store')
		#print('Check files: ', files)
		for f in files:
			filename= p + '/' + f 
			idx=filename
			dic[idx]={'text': None, 'truth': None, 'deps': None, 'sc': 0, 'wc':0, 'caps':0}
			#fake=0, real=1
			with open(filename, 'r') as infile:
				try:
					txt=infile.read().strip()
					raw= BeautifulSoup(txt, 'lxml').get_text()

					if len(raw)<=1:
						dic[idx]['text']= txt
					else:
						dic[idx]['text']= raw

					#Getting label
					if 'fake' in idx:
						dic[idx]['truth']= 0
					else:
						if 'real' in idx:
							dic[idx]['truth']= 1

					#Getting caps info for capitalization feature
					lazy_tokens= raw.split(' ')
					for lazy in lazy_tokens:
						if lazy.isupper():
							dic[idx]['caps']+=1

					#Start subprocesses		
					cfile= udSP(filename, t)
					deps, wc= get_ud(cfile)
					dic[idx]['deps']= deps
					dic[idx]['wc']= wc
			
				except:
					#This is for files from which text can't be extracted
					#Getting label
					if 'fake' in idx:
						dic[idx]['truth']= 0
					else:
						if 'real' in idx:
							dic[idx]['truth']= 1
					
					cfile= udSP(filename, t)
					deps, wc= get_ud(cfile)
					
					if deps!=[]:
						dic[idx]['deps']= deps
						dic[idx]['wc']= wc
						
						#in case BS doesn't recognize 
						fix=quickfix(cfile)
						if type(fix)==str:
							dic[idx]['text']= fix
							
							#get caps feature
							lazy_tokens= fix.split(' ')
							for lazy in lazy_tokens:
								if lazy.isupper():
									dic[idx]['caps']+=1
					else:
						del dic[idx]
	return dic


#######################
#For difficult-to-read text files
#######################

def quickfix(conll):
	print('ATTN: quick fix')
	with open(conll) as infile:
		infile= infile.read()
		infilesplit=infile.split('\n')
		text=[]
		for sentence in infilesplit:
			sentencesplit= sentence.split('\t')
			if len(sentencesplit) ==10:
				word= sentencesplit[1]
				text.append(word)
			else:
				pass
		text=' '.join(text)
	return text


#######################
#Reading the conll files and extracting all the UD labels (features)
#######################

def get_ud(conll):
	print('#######################')
	print('Getting UD features')

	#Account for formatting & differences between UD v1 and v2
	colons=['acl:relcl', 'aux:pass', 'cc:preconj', 'det:predet', 'csubj:pass', 'compound:prt', 'flat:foreign', 'flat:name', 'nmod:npmod', 'nmod:poss', 'nmod:tmod', 'nsubj:pass', 'obl:agent', 'obl:npmod', 'obl:tmod']
	nocolons=['aclrelcl', 'auxpass', 'ccpreconj', 'detpredet', 'csubjpass', 'compoundprt', 'flatforeign', 'flatname', 'nmodnpmod', 'nmodposs', 'nmodtmod', 'nsubjpass', 'oblagent', 'oblnpmod', 'obltmod']
	with open(conll) as infile:
		infile= infile.read()
		infilesplit=infile.split('\n')
		wc= len(infilesplit)-1
		a=[]
		deps=[]
		for sentence in infilesplit:
			sentencesplit= sentence.split('\t')
			if len(sentencesplit) ==10:
				feature= sentencesplit[7]
				if feature in nocolons:
					ix= nocolons.index(feature)
					feature=colons[ix]
				a.append(feature)
			else:
				if a!=[]:
					o=' '.join(a)
					print(o)
					deps.append(o)
				a=[]
		if deps==[]:
			deps=['root']
	return deps, wc


#######################
#Creating corpora
#######################

def makecorpora(traindic, testdic):
	print('#######################')
	print('Making corpora')
	ud_corpus=[]
	text_corpus=[]
	dics=[traindic, testdic]
	for dic in dics:
		for idx in dic:
			#bigram corpus
			if dic[idx]['text'] not in text_corpus:
				text_corpus.append(dic[idx]['text'])
			#UD corpus
			onedoc=[]
			if len(dic[idx]['deps'])==1:
				if dic[idx]['deps'][0] not in ud_corpus:
					ud_corpus.append(dic[idx]['deps'][0])
			elif len(dic[idx]['deps'])>1:
				onedoc= ' '.join(dic[idx]['deps'])
				if onedoc not in ud_corpus:
					ud_corpus.append(onedoc)
			else:
				pass
	ud_corpus=' '.join(ud_corpus)
	text_corpus=' '.join(text_corpus)
	return [ud_corpus], [text_corpus]



#######################
#Making the corpora
#Putting all the labels in both the training and test sets to avoid array size/shape errors
#######################


def makevectors(dic, ud_corpus, text_corpus):
	print('#######################')
	print('Making vectors')

	#the truth labels of each document
	labels=[1]

	#WC feature
	wc=[1]
	
	#SC feature
	sc=[1]

	#Capitalization feature
	c=[1]

	features= [('sc', sc), ('wc', wc), ('caps', c)]
	#This accounts for the number of sentences in each doc
	for idx in dic:
		#Getting the SC feature
		idxsc= len(dic[idx]['deps'])
		dic[idx]['sc']= idxsc

		#Getting lexical features
		for feature in features:
			feature[1].append(dic[idx][feature[0]])

		#Getting label 
		labels.append(dic[idx]['truth'])

		#Creating the text corpus for the bigram feature
		text_corpus.append(dic[idx]['text'])

		#Creating the corpus that replaces the words with UD labels
		if len(dic[idx]['deps'])==1:
			ud_corpus.append(dic[idx]['deps'][0])
		elif len(dic[idx]['deps'])>1:
			onedoc= ' '.join(dic[idx]['deps'])
			ud_corpus.append(onedoc)
		else:
			pass

	print('#######################')
	print('Starting vectors')

	#count vector for ud
	udv= CountVectorizer(analyzer='word', token_pattern= r'\w+\:\w+|\w+')
	ud= udv.fit_transform(ud_corpus).toarray()

	#bigram vector
	bv= CountVectorizer(analyzer='word', ngram_range=(2,2))
	bigrams= bv.fit_transform(text_corpus).toarray()

	#reshaping the lexical features
	features= [sc, wc, c]
	n=len(wc) #number of arrays
	for feature in features:
		feature= np.asarray(feature)
		feature= feature.reshape((wcl,1))

	#putting all the features together
	ub=np.hstack((ud, bigrams))
	ubw=np.hstack((ub, wc))
	ubws=np.hstack((ubw, sc))
	all_features=np.hstack((ubws, c))

	#labeling of fake and real for the whole docs
	labels = np.array(labels)

	#getting the feature names
	all_featurenames=[]
	#UD
	ud_names= udv.get_feature_names()
	all_featurenames=ud_names
	#bigrams
	for b in bv.get_feature_names():
		all_featurenames.append(b)
	#word count
	all_featurenames.append('wc')
	#sentence count
	all_featurenames.append('sc')
	#capitalization
	all_featurenames.append('caps')

	return all_features, labels, all_featurenames


def ml(counts, labels):
	clf= svm.SVC(kernel='linear')
	clf.fit(counts, labels)
	return clf


def predict(clf, test_counts, test_labels):
	predictions= clf.predict(test_counts)
	accuracy= accuracy_score(test_labels, predictions)
	#print('##################################')
	#print('Predictions')
	#print(predictions)
	print('##################################')
	print('Accuracy:', accuracy)
	print()
	print('Classification report')
	print(classification_report(test_labels, predictions))
	print('##################################')

    

def show_most_informative_features(clf, all_featurenames, n=20):
	tvec= clf.coef_
	coefs= sorted(zip(tvec[0], all_featurenames), key=itemgetter(0), reverse=True)
	topn  = zip(coefs[:n], coefs[:-(n+1):-1])
	output=[]
	# Create two columns with most negative and most positive features.
	for (cp, fnp), (cn, fnn) in topn:
		output.append("{:0.4f}{: >15}    {:0.4f}{: >15}".format(cp, fnp, cn, fnn))
	return "\n".join(output)


def plot_coefficients(classifier, feature_names, top_features=4):
	rcParams.update({'figure.autolayout': True})
    coef = classifier.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(15, 5))
    colors = ['black' if c < 0 else 'grey' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    print('top features', top_features)
    print('feature_names.shape', feature_names.shape)
    print('top_coefficients', top_coefficients)
    print('top_coefficients shape', top_coefficients.shape)
    plt.xticks(np.arange(1, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.show()




print()
print('###################')
print('Making corpora')


#directories of training/test docs
traindir= os.getcwd() + '/train'
testdir= os.getcwd() + '/test'

#Create dictionaries
traindic=bf(traindir, t='train')
testdic=bf(testdir, t='test')

#Create corpora 
ud_corpus, text_corpus=makecorpora(traindic, testdic)

#Training vectors
training, training_labels, training_featurenames=makevectors(traindic, ud_corpus, text_corpus)

#Corpora
ud_corpus, text_corpus=makecorpora(traindic, testdic)

#Testing vectors
testing, testing_labels, testing_featurenames=makevectors(testdic, ud_corpus, text_corpus)


print('###################')
print('Training')
clf=ml(training, training_labels)


print('###################')
print('Predicting')
predictions= predict(clf, testing, testing_labels)
print(predictions)


print('###################')
print('Most informative features')
mif=show_most_informative_features(clf, training_featurenames, n=40)
print('Most informative features')
print(mif)

#Plot informative features
plot_coefficients(clf, training_featurenames, top_features=10)


timeElapsed=datetime.now()-startTime 
print('###################')
print('Time elapsed (hh:mm:ss.ms) {}'.format(timeElapsed))