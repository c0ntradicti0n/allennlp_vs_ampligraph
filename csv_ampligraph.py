import logging
from argparse import ArgumentParser
from collections import OrderedDict
import numpy as np
import pandas as pd
from ampligraph.datasets import load_wn18
from ampligraph.latent_features import ComplEx, HolE, TransE
from ampligraph.evaluation import evaluate_performance, mrr_score, hits_at_n_score
from ampligraph.latent_features import ComplEx
from ampligraph.utils import save_model, restore_model
import os
import tensorflow as tf
import random
from numpy import cumsum
from more_itertools import flatten
from sklearn.utils import Memory
import pprint
from tspy import TSP
import numpy as np
from pandas import CategoricalDtype
from scipy.spatial.distance import cdist



logging.getLogger().setLevel(logging.INFO)
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

parser = ArgumentParser(description='Projecting graph to 3d (and embeddings)')
parser.add_argument('csv',
                    nargs='?',
                    type=str,
                    help='csv with n1, n2, rel columns',
                    default="./test")
args = parser.parse_args()


# getting whole wordnet graph
ke_model_path = "./knowledge_graph_model/csv_ke.amplimodel"
ke_wnkeys_path = "./knowledge_graph_model/csv_ke.wnkeys"

table = pd.read_csv(args.csv, sep='|', header=0)
whole_graph = list(zip(table['n1'], table['rel'], table['n2']))

if True: #not os.path.isfile(ke_wnkeys_path) or not os.path.isfile(ke_model_path):
    pprint.pprint (whole_graph[:60])
    random.shuffle(whole_graph)


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
    X = percentage_split(whole_graph, corpus_split_layout)
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

    model = TransE(verbose=True, k=70, epochs=300)

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


    model2 = TransE(verbose=True, k=3, epochs=300)
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

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

print("Extracting Embeddings..")
alle = table['n1'].tolist() + table['n2'].tolist()
embedding_map = dict([(str(a), (model.get_embeddings(str(tok2id[str(a)])), tok2id[str(a)])) for a in alle if str(a) in tok2id])
embedding_map2 = dict([(str(a), (model2.get_embeddings(str(tok2id[str(a)])), tok2id[str(a)])) for a in alle if str(a) in tok2id])

embeddings_array = np.array([i[0] for i in embedding_map.values()])
print ("PCA")
embeddings_3d_pca = PCA(n_components=3).fit_transform(embeddings_array)
print ("TSNE")
embeddings_3d_tsne = TSNE(n_components=3).fit_transform(embeddings_array)
print("k2")
embeddings_k2 = np.array([i[0] for i in embedding_map2.values()])

print (embeddings_3d_pca.shape)
print (embeddings_k2.shape)

print ("pandas")
table = pd.DataFrame(data={'name':list(s.replace("Synset('", '').replace("')", "") for s in embedding_map.keys()),
                           'id': [i[1] for i in embedding_map.values()],
                           'x_pca': embeddings_3d_pca[:, 0],
                           'y_pca': embeddings_3d_pca[:, 1],
                           'z_pca': embeddings_3d_pca[:, 2],
                           'x_tsne': embeddings_3d_tsne[:, 0],
                           'y_tsne': embeddings_3d_tsne[:, 1],
                           'z_tsne': embeddings_3d_tsne[:, 2],
                           'x_k2': embeddings_k2[:, 0],
                           'y_k2': embeddings_k2[:, 1],
                           'z_k2': embeddings_k2[:, 2]
                           })

print ('clusters')
import hdbscan
std_args = {
'algorithm':'best',
    'alpha':1.0,
    'approx_min_span_tree':True,
    'gen_min_span_tree':False,
    'leaf_size':20,
    'memory': Memory(cachedir=None),
    'metric':'euclidean',
    'min_cluster_size':13,
    'min_samples':None,
    'p':None
}

def cluster(embeddings_array, **kwargs):
    print ('dimensionality', embeddings_array.shape)
    clusterer = hdbscan.HDBSCAN(**kwargs)
    clusterer.fit(np.array(embeddings_array))
    print ('number of clusters: ', max(clusterer.labels_))
    return clusterer.labels_

table['cl_pca'] =  cluster(embeddings_3d_pca, **std_args)
table['cl_tsne'] = cluster(embeddings_3d_tsne, **std_args)
table['cl_k2'] =   cluster(embeddings_k2, **std_args)
table['cl_kn'] =   cluster(embeddings_array, **std_args)

table.to_csv("./knowledge_graph_coords/knowledge_graph_3d_choords.csv", sep='\t', header=True,
                                                                  index=False)
table = pd.read_csv("./knowledge_graph_coords/knowledge_graph_3d_choords.csv", index_col=0, sep='\t')
things = ['pca', 'tsne', 'k2', 'kn']

def make_path (X, D):
    tsp = TSP()
    # Using the data matrix
    tsp.read_data(X)

    # Using the distance matrix
    tsp.read_mat(D)

    from tspy.solvers import TwoOpt_solver
    two_opt = TwoOpt_solver(initial_tour='NN', iter_num=100000)
    two_opt_tour = tsp.get_approx_solution(two_opt)

    #tsp.plot_solution('TwoOpt_solver')

    best_tour = tsp.get_best_solution()
    return best_tour

for kind in things:
    print ("writing table for %s " % kind)
    table['cl'] = table['cl_%s' % kind]
    cl_cols = table[['cl_%s' % k for k in things]]
    cl_df = table.groupby(by='cl').mean().reset_index()

    # Initialize fitness function object using coords_list
    print ("optimizing the path through all centers")
    if kind == "kn":
        subkind = "tsne"
    else:
        sub_kind = kind

    subset = cl_df[[c + "_" + sub_kind for c in ['x', 'y', 'z']]]
    print (subset[:10])

    points = [list(x) for x in subset.to_numpy()]
    print (points[:10])
    print (len(points))

    arr = np.array(points)
    dist = Y = cdist(arr, arr, 'euclidean')
    new_path = make_path(np.array(points), dist)[:-1]
    print (new_path)

    cl_df[['cl_%s' % k for k in things]] = cl_cols


    path_order_categories = CategoricalDtype(categories=new_path,  ordered = True)
    cl_df['cl_%s' % kind] = cl_df['cl'].astype(path_order_categories)

    cl_df.sort_values(['cl_%s' % kind], inplace=True)
    cl_df['cl_%s' % kind] = cl_df['cl'].astype('int32')

    cl_df.to_csv('./knowledge_graph_coords/%s_clusters_mean_points.csv' % kind, sep='\t', header=True,
                                                                  index=False)
    print (kind + " " + str(new_path))

logging.info("ampligraph and clustering finished")