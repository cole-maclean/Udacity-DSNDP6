from bs4 import BeautifulSoup
import urllib
import feedparser
import time
import csv
from random import random
#parse table of categories from arxiv into lookup table for {discipline:[categories]}
def build_arxiv_cat_index():
	url = 'http://arxiv.org/help/api/user-manual#python_simple_example'
	data = urllib.urlopen(url).read()
	soup = BeautifulSoup(data, 'html.parser')
	cat_table = soup.findAll('tbody')[7]
	results = {}
	for row in cat_table.findAll('tr'):
	    aux = row.findAll('td')
	    try:
	        if aux[0].string.strip().split('.')[0] in results.keys():
	            results[aux[0].string.strip().split('.')[0]].append(aux[0].string.strip())
	        else:
	            results[aux[0].string.strip().split('.')[0]] =[aux[0].string.strip()]
	    except AttributeError:
	        pass
	return results

#loop through arxiv query for category and return results
def query_arxiv(category):
	url = 'http://export.arxiv.org/api/query?search_query=cat:' + category + '&max_results=1'
	data = urllib.urlopen(url).read()
	d = feedparser.parse(data)
	max_results =  d['feed']['opensearch_totalresults']
	query_start = 1
	query_results = ''
	print category
	while query_start < int(max_results):
		url = ('http://export.arxiv.org/api/query?search_query=cat:'
		+ category + '&max_results=3000&start=' + str(query_start) + '&sortBy=submittedDate&sortOrder=ascending')
		query_results = query_results + urllib.urlopen(url).read()
		query_start = query_start + 3000
		print round(float(query_start)/float(max_results),1)*100
		time.sleep(3)
	return query_results

#http://stackoverflow.com/questions/27889873/clustering-text-documents-using-scikit-learn-kmeans-in-python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans
import string
from nltk import word_tokenize
from nltk.stem import PorterStemmer
import difflib

#paper categorization model
def paper_category_model(paper_titles):
    true_k = 10
    my_words = ['based','mems','using','non','use','models','spaces','type'] #custom stop words
    my_stop_words = set(text.ENGLISH_STOP_WORDS.union(my_words))
    vectorizer = TfidfVectorizer(stop_words=my_stop_words,max_features=8)
    X = vectorizer.fit_transform(paper_titles)
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,random_state=42)
    model.fit(X)
    return (vectorizer,model)

#find best parent category based on simple word differences in category labels
def get_best_parent(paper_category,sorted_categories,discipline):
    bp = difflib.get_close_matches(paper_category,sorted_categories,n=1,cutoff=0.2)
    if bp:
    	s = difflib.SequenceMatcher(None, paper_category, bp[0])
        return bp[0], s.ratio()
    else:
        return discipline, 0

import copy
import json
import os

#function to parse papers into visualization data model
def build_data_model(results):
	sorted_entries = sorted(results,key= lambda k:k['published'][0:4])
	cat_list = ['stat','cs','math','physics']
	print "data sorted"
	old_year = int(sorted_entries[0]['published'][0:4])
	data_model = {old_year:{}}
	all_paper_titles ={}
	discipline_cat_models = {} #0 - vectorizer, 1-model
	for paper in sorted_entries:
		discipline = paper['arxiv_primary_category']['term'].split('.')[0]
		if discipline in cat_list:
		    new_year = int(paper['published'][0:4])
		    paper_title = paper['title']
		    if discipline in discipline_cat_models.keys():
		        disc_vectorizer,disc_model = discipline_cat_models[discipline]
		        paper_vector = disc_vectorizer.transform([paper['summary']])
		        paper_cluster = disc_model.predict(paper_vector)[0]
		        ordered_centroids = disc_model.cluster_centers_.argsort()[:, ::-1]
		        terms = disc_vectorizer.get_feature_names()
		        try:
		            paper_category = ' '.join(sorted([terms[ind] for ind in ordered_centroids[paper_cluster, :3]]))
		        except IndexError:
		            paper_category = discipline
		    else:
		        paper_category = discipline
		    if new_year != old_year:
		    	print new_year
		        data_model[new_year] = copy.deepcopy(data_model[old_year])#stores results of previous year to build cumulative model
		        old_year = new_year
		        for model_discipline in all_paper_titles.keys():
		            if len(all_paper_titles[model_discipline]) > 50:
		                discipline_cat_models[model_discipline] = paper_category_model(all_paper_titles[model_discipline])
		    parsed_paper = {k:v for (k,v) in paper.iteritems()
								 if k in ['title','author','link']}  
		    if discipline in data_model[new_year].keys():
		        sorted_categories = ([k for (k,v) in sorted(data_model[new_year][discipline]['categories'].items(),
		                                                    key=lambda (k, v): v['paper_count'],reverse=True)])
		        best_parent, diff_ratio = get_best_parent(paper_category,sorted_categories,discipline)
		        if diff_ratio >= 0.7:
		            paper_category = best_parent
		        all_paper_titles[discipline].append(paper_title)
		        if paper_category in data_model[new_year][discipline]['categories'].keys():
		            data_model[new_year][discipline]['paper_count']+=1
		            data_model[new_year][discipline]['categories'][paper_category]['papers'].append(parsed_paper)
		            data_model[new_year][discipline]['categories'][paper_category]['paper_count'] +=1
		        else:
		            data_model[new_year][discipline]['paper_count']+=1
		            data_model[new_year][discipline]['categories'][paper_category] = ({'parent_cat':best_parent,
		                                                                               'category':paper_category,
		                                                                               'papers':[parsed_paper],
		                                                                              'paper_count':1})
		    else:
		        all_paper_titles[discipline]=[paper_title]
		        best_parent = discipline
		        data_model[new_year][discipline]=({'paper_count':1,
		            'categories':{paper_category:{'parent_cat':best_parent,'category':paper_category,
		                                            'papers':[parsed_paper],'paper_count':1}}})
	return data_model

def store_data_model(data_model):
	for yr in data_model.keys():
	    for disc in data_model[yr].keys():
	        sorted_categories = ([k for (k,v) in sorted(data_model[yr][disc]['categories'].items(),
	                                                    key=lambda (k, v): v['parent_cat'],reverse=False)])
	        data_model[yr][disc]['children'] = ([{k:v for (k,v) in data_model[yr][disc]['categories'][category].iteritems()}
	                                              for category in sorted_categories])
	        data_model[yr][disc]['parents'] = list(set([k['parent_cat'] for k in data_model[yr][disc]['children']]))

	with open('C:/Users/Cole/Desktop/Udacity/Data Analyst Nano Degree/Project 6/Udacity-DSNDP6/disciplines.json', 'w') as myfile:
	    disc_list = sorted(['stat','cs','math','physics'])
	    json.dump(disc_list, myfile)
	    print 'success!'
	        
	with open('C:/Users/Cole/Desktop/Udacity/Data Analyst Nano Degree/Project 6/Udacity-DSNDP6/data_model.json', 'w') as fp:
	    json.dump(data_model, fp)
	    print 'success!!'

def append_record(record):
    with open('C:/Users/Cole/Desktop/Udacity/Data Analyst Nano Degree/Project 6/Udacity-DSNDP6/arxiv_data.json', 'a') as f:
        json.dump(record, f)
        f.write(os.linesep)

def load_arxiv_data():
	arxiv_cat_index = build_arxiv_cat_index()
	cat_list = ['stat','cs','math','cond-mat','physics']
	for discipline in cat_list:
		for category in arxiv_cat_index[discipline]:
			results = query_arxiv(category)
			parsed_results =feedparser.parse(results)['entries']
			parsed_papers = ([{k:v for (k,v) in paper.iteritems()
								 if k not in ['published_parsed','summary_detail','updated_parsed']}
								 for paper in parsed_results])
			for paper in parsed_papers:
				append_record(paper)
def main():
	arxiv_data =[]
	with open('C:/Users/Cole/Desktop/Udacity/Data Analyst Nano Degree/Project 6/arxiv_data.json') as f:
		for line in f:
			if random() <= .01:
				arxiv_data.append(json.loads(line))
		print "data loaded"
	data_model = build_data_model(arxiv_data)
	store_data_model(data_model)
	print 'success!!!'

#load_arxiv_data()
main()
