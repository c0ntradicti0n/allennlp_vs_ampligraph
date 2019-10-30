from collections import OrderedDict

import numpy as np
import pandas as pd
from ampligraph.datasets import load_wn18
from ampligraph.latent_features import ComplEx, HolE, TransE
from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score

from ampligraph.latent_features import ComplEx

import os
from ampligraph.utils import save_model, restore_model

import tensorflow as tf
from more_itertools import flatten
from nltk.corpus.reader import Synset, Lemma
from ordered_set import OrderedSet

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from ampligraph.evaluation import evaluate_performance


# getting whole wordnet graph
ke_model_path = "./knowledge_graph_model/wikirr_ke.amplimodel"
ke_wnkeys_path = "./knowledge_graph_model/wikirr_ke.wnkeys"

from wordnet2relationmapping import get_synonyms, get_antonyms, get_hypernyms, get_cohyponyms, get_cohypernyms, \
    get_hyponyms, alle

from nltk.corpus import wordnet as wn

wordnet_getters = {
    #"synonym": get_synonyms,
    "antonym": get_antonyms,
    "hypernym": get_hypernyms,
    # "cohyponym":get_cohyponyms,
    # "cohypernym":get_cohypernyms,
    "hyponym": get_hyponyms
}

if not os.path.isfile(ke_wnkeys_path) or not os.path.isfile(ke_model_path):


    all_synsets = wn.all_synsets()

    def wordnet_edges (synset):
        for rel, fun in wordnet_getters.items():
            res = fun(synset)
            if isinstance(res, Synset):
                raise ValueError('synset')
            for related in res:
                for lemma in synset.lemmas():
                    if isinstance(related, Synset):
                        for rel_lemma in related.lemmas():
                            if lemma != rel_lemma:
                                yield list((str(lemma.synset()), rel, str(rel_lemma.synset())))
                    else:
                        if lemma != related:
                            yield list((str(lemma.synset()), rel, str(related.synset())))

    whole_wn_graph = [list(wordnet_edges(syn)) for syn in all_synsets]
    print ("size of the whole graph: ", len(whole_wn_graph))
    whole_wn_graph = list(flatten(whole_wn_graph)) # [: int(len(whole_wn_graph) *0.7)]
    known_rels = OrderedSet([str(rel) for rel in whole_wn_graph])
    whole_wn_graph = [rel for rel in whole_wn_graph if str(rel) in known_rels]

    import pprint
    pprint.pprint (whole_wn_graph[:60])

    import random
    random.shuffle(whole_wn_graph)


    from numpy import cumsum
    def percentage_split(seq, percentage_dict):

        cdf = cumsum(list(percentage_dict.values()))
        assert cdf[-1] == 1.0
        stops = list(map(int, cdf * len(seq)))
        return {key: seq[a:b] for a, b, key in zip([0]+stops, stops, percentage_dict.keys())}


    corpus_split_layout = {
        'train': 0.8,
        'test': 0.1,
        'valid': 0.1
    }
    X = percentage_split(whole_wn_graph, corpus_split_layout)
    known_entities = set (flatten([r[0], r[2]] for r in X['train']))


    id2tok = {i:tok for i, tok in enumerate(known_entities)}
    tok2id = {tok:i for i, tok in enumerate(known_entities)}

    import pickle
    with open(ke_wnkeys_path, 'wb') as handle:
        pickle.dump((tok2id, id2tok), handle)

    X['train'] = np.array([list((tok2id[r[0]], r[1], tok2id[r[2]])) for r in X['train'] if r[0] in known_entities and r[2] in known_entities])

    X['valid'] = np.array([list((tok2id[r[0]], r[1], tok2id[r[2]])) for r in X['valid'] if r[0] in known_entities and r[2] in known_entities])
    X['test'] = np.array([list((tok2id[r[0]], r[1], tok2id[r[2]])) for r in X['test'] if r[0] in known_entities and r[2] in known_entities])

    #import guppy
    #h = guppy.hpy()
    #print (h.heap())

    X_train, X_valid = X['train'], X['valid']
    print('Train set size: ', X_train.shape)
    print('Test set size: ', X_valid.shape)

    """
    k=DEFAULT_EMBEDDING_SIZE,
    eta=DEFAULT_ETA,
    epochs=DEFAULT_EPOCH,
    batches_count=DEFAULT_BATCH_COUNT,
    seed=DEFAULT_SEED,
    embedding_model_params={'norm': DEFAULT_NORM_TRANSE,
                         'normalize_ent_emb': DEFAULT_NORMALIZE_EMBEDDINGS,
                         'negative_corruption_entities': DEFAULT_CORRUPTION_ENTITIES,
                         'corrupt_sides': DEFAULT_CORRUPT_SIDE_TRAIN},
    optimizer=DEFAULT_OPTIM,
    optimizer_params={'lr': DEFAULT_LR},
    loss=DEFAULT_LOSS,
    loss_params={},
    regularizer=DEFAULT_REGULARIZER,
    regularizer_params={},
    initializer=DEFAULT_INITIALIZER,
    initializer_params={'uniform': DEFAULT_XAVIER_IS_UNIFORM},
    verbose=DEFAULT_VERBOSE):
    """

    model = TransE(verbose=True, k=70, epochs=40)

    """
    model = ComplEx(batches_count=10, seed=0, epochs=60, k=50, eta=10,
                    # Use adam optimizer with learning rate 1e-3
                    optimizer='adam', optimizer_params={'lr': 1e-3},
                    # Use pairwise loss with margin 0.5
                    loss='pairwise', loss_params={'margin': 0.5},
                    # Use L2 regularizer with regularizer weight 1e-5
                    regularizer='LP', regularizer_params={'p': 2, 'lambda': 1e-5},
                    # Enable stdout messages (set to false if you don't want to display)
                    verbose=True)"""



    print ("Training...")
    x_orig = load_wn18()
    model.fit(X_train)


    save_model(model, model_name_path=ke_model_path)


    model2 = TransE(verbose=True, k=3, epochs=40)
    model2.fit(X_train)
    save_model(model2, model_name_path=ke_model_path + '2')

    #filter_triples = np.concatenate((X_train, X_valid))
    #filter = np.concatenate((X['train'], X['valid'], X['test']))
    #ranks = evaluate_performance(X['test'],
    #                             model=model,
    #                             filter_triples=filter,
    #                             use_default_protocol=True,  # corrupt subj and obj separately while evaluating
    #                             verbose=True)

    #mrr = mrr_score(ranks)
    #hits_10 = hits_at_n_score(ranks, n=10)
    #print("MRR: %f, Hits@10: %f" % (mrr, hits_10))
    # Output: MRR: 0.886406, Hits@10: 0.935000
else:
    model = restore_model(model_name_path=ke_model_path)
    model2 = restore_model(model_name_path=ke_model_path+'2')

    import pickle

    with open(ke_wnkeys_path, 'rb') as handle:
        tok2id, id2tok = pickle.load(handle)

import pprint

def find_in_tok2id(w):
    for s in tok2id.keys():
        if w in s:
            print (w, s, "it is alphabetically there")

tok2id = OrderedDict (tok2id)
def predict (w1,w2):
    try:
        rellist = [[tok2id[str(s1)], k, tok2id[str(s2)]] for s1 in wn.synsets(w1) for s2 in wn.synsets(w2)   for k in wordnet_getters.keys() if str(s1) in tok2id and str(s2) in tok2id]
        prediction = model.predict(np.array(rellist))
        sorted_predictions = sorted(list(zip([p[0] for p in prediction], [r[1] for r in rellist])), key=lambda x:x[0])
        print (w1, w2)
        pprint.pprint (sorted_predictions[:3])
        pprint.pprint (sorted_predictions[-3:])
    except KeyError:
        print (w1, w2, " is oov")
        find_in_tok2id(w1)
        find_in_tok2id(w2)




    #for val, task in zip(prediction, rellist):s
    #    print ("%s $ for %s" % (str(val[0]), task[1]))


w1 = 'good'
w2 = 'bad'
predict(w1, w2)


w1 = 'true'
w2 = 'false'
predict(w1, w2)


w1 = 'tree'
w2 = 'plant'
predict(w1, w2)

w1 = 'spoon'
w2 = 'kitchen'
predict(w1, w2)

w1 = 'fish'
w2 = 'shark'
predict(w1, w2)

w1 = 'dog'
w2 = 'bark'
predict(w1, w2)

w1 = 'black'
w2 = 'white'
predict(w1, w2)

w1 = 'animal'
w2 = 'dog'
predict(w1, w2)

w1 = 'radio'
w2 = 'podcast'
predict(w1, w2)


w1 = 'god'
w2 = 'devil'
predict(w1, w2)


w1 = 'wood'
w2 = 'tree'
predict(w1, w2)


w2 = 'wood'
w1 = 'tree'
predict(w1, w2)

w1 = 'light'
w2 = 'shadow'
predict(w1, w2)

w1 = 'street'
w2 = 'rue'
predict(w1, w2)

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

print("Extracting Embeddings..")

embedding_map = dict([(str(a), (model.get_embeddings(str(tok2id[str(a)])), tok2id[str(a)])) for a in alle if str(a) in tok2id])
embedding_map2 = dict([(str(a), (model2.get_embeddings(str(tok2id[str(a)])), tok2id[str(a)])) for a in alle if str(a) in tok2id])

embeddings_array = np.array([i[0] for i in embedding_map.values()])
print ("PCA")
embeddings_3d_pca = PCA(n_components=3).fit_transform(embeddings_array)
print ("TSNE")
embeddings_3d_tsne = TSNE(n_components=3).fit_transform(embeddings_array)
print("k=2")
embeddings_k2 = np.array([i[0] for i in embedding_map2.values()])

print (embeddings_3d_pca.shape)
print (embeddings_k2.shape)

print ("pandas")
table = pd.DataFrame(data={'name':list(embedding_map.keys()),
                           'id': [i[1] for i in embedding_map.values()],
                           'x_pca': embeddings_3d_pca[:, 0],
                           'y_pca': embeddings_3d_pca[:, 1],
                           'z_pca': embeddings_3d_pca[:, 2],
                           'x_tsne': embeddings_3d_tsne[:, 0],
                           'y_tsne': embeddings_3d_tsne[:, 1],
                           'z_tsne': embeddings_3d_tsne[:, 2],
                           'x_k=2': embeddings_k2[:, 0],
                           'y_k=2': embeddings_k2[:, 1],
                           'z_k=2': embeddings_k2[:, 2]
                           })

table.to_csv("knowledge_graph_3d_choords.csv")

plot_clusters("cluster")


from sklearn import metrics
metrics.adjusted_rand_score(plot_df.continent, plot_df.cluster)

df["results"] = (df.home_score > df.away_score).astype(int) + \
                (df.home_score == df.away_score).astype(int)*2 + \
                (df.home_score < df.away_score).astype(int)*3 - 1

df.results.value_counts(normalize=True)


def get_features_target(mask):
    def get_embeddings(team):
        return team_embeddings.get(team, np.full(list(team_embeddings.values())[0].shape[0], np.nan))

    X = np.hstack((np.vstack(df[mask].home_team_id.apply(get_embeddings).values),
                   np.vstack(df[mask].away_team_id.apply(get_embeddings).values)))
    y = df[mask].results.values
    return X, y

clf_X_train, y_train = get_features_target((df["train"]))
clf_X_test, y_test = get_features_target((~df["train"]))

clf_X_train.shape, clf_X_test.shape

np.isnan(clf_X_test).sum()/clf_X_test.shape[1]

from xgboost import XGBClassifier

clf_model = XGBClassifier(n_estimators=550, max_depth=5, objective="multi:softmax")

clf_model.fit(clf_X_train, y_train)

print (df[~df["train"]].results.value_counts(normalize=True))

print (metrics.accuracy_score(y_test, clf_model.predict(clf_X_test)))




