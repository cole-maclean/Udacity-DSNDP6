from bs4 import BeautifulSoup
import urllib
import feedparser
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
	           + category + '&max_results=1&start=' + str(query_start) + '&sortBy=submittedDate&sortOrder=ascending')
	    query_results = query_results + urllib.urlopen(url).read()
	    query_start = query_start + 100
	return query_results

#http://stackoverflow.com/questions/27889873/clustering-text-documents-using-scikit-learn-kmeans-in-python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans
import difflib

def paper_category_model(paper_titles):
    true_k = 10
    my_words = ['based','mems','using']
    my_stop_words = set(text.ENGLISH_STOP_WORDS.union(my_words))
    vectorizer = TfidfVectorizer(stop_words=my_stop_words,min_df=0.01)
    X = vectorizer.fit_transform(paper_titles)
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1,random_state=42)
    model.fit(X)
    return (vectorizer,model)

def get_best_parent(paper_category,sorted_categories,discipline):
    bp = difflib.get_close_matches(paper_category,sorted_categories,1)
    if bp:
        return bp[0]
    else:
        return discipline

import copy
import json

def build_data_model(results):
	sorted_entries = sorted(results,key= lambda k:k['published'][0:4])
	old_year = int(sorted_entries[0]['published'][0:4])
	data_model = {old_year:{}}
	all_paper_titles ={}
	discipline_cat_models = {} #0 - vectorizer, 1-model
	limit_idx =0
	for paper in sorted_entries:
		if limit_idx%1==0:
		    new_year = int(paper['published'][0:4])
		    discipline = paper['arxiv_primary_category']['term'].split('.')[0]
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
		        data_model[new_year] = copy.deepcopy(data_model[old_year])#stores results of previous year to build cummulative model
		        old_year = new_year
		        for model_discipline in all_paper_titles.keys():
		            if len(all_paper_titles[model_discipline]) > 20:
		                discipline_cat_models[model_discipline] = paper_category_model(all_paper_titles[model_discipline])  
		    if discipline in data_model[new_year].keys():
		        sorted_categories = ([k for (k,v) in sorted(data_model[new_year][discipline]['categories'].items(),
		                                                    key=lambda (k, v): v['paper_count'],reverse=True)])
		        all_paper_titles[discipline].append(paper_title)
		        if paper_category in data_model[new_year][discipline]['categories'].keys():
		            data_model[new_year][discipline]['paper_count']+=1
		            data_model[new_year][discipline]['categories'][paper_category]['papers'].append(paper)
		            data_model[new_year][discipline]['categories'][paper_category]['paper_count'] +=1
		        else:
		            best_parent = get_best_parent(paper_category,sorted_categories,discipline)
		            data_model[new_year][discipline]['paper_count']+=1
		            data_model[new_year][discipline]['categories'][paper_category] = ({'parent_cat':best_parent,
		                                                                               'category':paper_category,
		                                                                               'papers':[paper],
		                                                                              'paper_count':1})
		    else:
		        all_paper_titles[discipline]=[paper_title]
		        best_parent = discipline
		        data_model[new_year][discipline]=({'paper_count':1,
		            'categories':{paper_category:{'parent_cat':best_parent,'category':paper_category,
		                                            'papers':[paper],'paper_count':1}}})
		limit_idx+=1
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
	    disc_list = data_model[max(data_model.keys())].keys()
	    json.dump(disc_list, myfile)
	    print 'success!'
	        
	with open('C:/Users/Cole/Desktop/Udacity/Data Analyst Nano Degree/Project 6/Udacity-DSNDP6/data_model.json', 'w') as fp:
	    json.dump(data_model, fp)
	    print 'success!!'

def load_arxiv_data():
	arxiv_cat_index = build_arxiv_cat_index()
	with open('C:/Users/Cole/Desktop/Udacity/Data Analyst Nano Degree/Project 6/Udacity-DSNDP6/arxiv_data.json', 'w') as myfile:
		json.dump([],myfile)
	for discipline in arxiv_cat_index.keys():
		for category in arxiv_cat_index[discipline]:
			results = query_arxiv(category)
			parsed_results =feedparser.parse(results)['entries']
			parsed_papers = ([{k:v for (k,v) in paper.iteritems()
								 if k not in ['published_parsed','summary_detail','updated_parsed']}
								 for paper in parsed_results])
			with open('C:/Users/Cole/Desktop/Udacity/Data Analyst Nano Degree/Project 6/Udacity-DSNDP6/arxiv_data.json', 'r') as myfile:
				feed = json.load(myfile)
			with open('C:/Users/Cole/Desktop/Udacity/Data Analyst Nano Degree/Project 6/Udacity-DSNDP6/arxiv_data.json', 'w') as myfile:
				if feed:
					for paper in parsed_papers:
						feed.append(paper)
					json.dump(feed, myfile)
				else:
					json.dump(parsed_papers, myfile)

def main():
	with open('C:/Users/Cole/Desktop/Udacity/Data Analyst Nano Degree/Project 6/Udacity-DSNDP6/arxiv_data.json', 'r') as myfile:
	    arxiv_data =json.load(myfile)
	data_model = build_data_model(arxiv_data)
	store_data_model(data_model)
	print 'success!!!'

#load_arxiv_data()
main()
