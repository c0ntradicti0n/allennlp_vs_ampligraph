
import numpy as np

from pandas import CategoricalDtype
from scipy.spatial.distance import cdist

import pandas

table = pandas.read_csv("knowledge_graph_3d_choords.csv", index_col=0)
things = ['pca', 'tsne', 'k2', 'kn']

from tspy import TSP


def make_path (X, D):
    tsp = TSP()
    # Using the data matrix
    tsp.read_data(X)

    # Using the distance matrix
    tsp.read_mat(D)

    from tspy.solvers import TwoOpt_solver
    two_opt = TwoOpt_solver(initial_tour='NN', iter_num=10000)
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

    cl_df.to_csv('%s_clusters_mean_points.csv' % kind, sep='\t', header=True,
                                                                  index=False)
    print (kind + " " + str(new_path))

