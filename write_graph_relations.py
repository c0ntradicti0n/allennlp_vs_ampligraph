import pprint

from more_itertools import flatten
from nltk.corpus.reader import Synset
import pandas as pd
from nltk.corpus import wordnet as wn
import numpy as np

from wordnet2relationmapping import get_synonyms, get_antonyms, get_hypernyms, get_cohyponyms, get_cohypernyms, \
    get_hyponyms, alle

wordnet_getters = {
    #"synonym": get_synonyms,
    "antonym": get_antonyms,
    "hypernym": get_hypernyms,
    # "cohyponym":get_cohyponyms,
    # "cohypernym":get_cohypernyms,
    "hyponym": get_hyponyms
}

all_synsets = wn.all_synsets()

def wordnet_edges (synset):
    for rel, fun in wordnet_getters.items():
        res = fun(synset)
        if isinstance(res, Synset):
            raise ValueError('synset')
        for related in res:
            for lemma in synset.lemmas():
                if isinstance(lemma, Synset):
                    n1 = lemma.name()
                else:
                    n1 = lemma.synset().name()
                if isinstance(related, Synset):
                    n2 = related.name()
                else:
                    n2 = related.synset().name()
                yield list([n1, rel, n2])


whole_wn_graph = [list(wordnet_edges(syn)) for syn in all_synsets]
whole_wn_graph = list(flatten(whole_wn_graph))
print ("size of the whole graph: ", len(whole_wn_graph))
pprint.pprint(whole_wn_graph[:60])
whole_graph_array = np.array(whole_wn_graph)
print ("Shape ", whole_graph_array.shape)

table = pd.DataFrame(data={'n1':   whole_graph_array[:,0],
                           'rel':  whole_graph_array[:,1],
                           'n2':   whole_graph_array[:,2]
                           })

table.to_csv("knowledge_graph_rels.csv")
