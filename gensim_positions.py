
import numpy as np
import pandas as pd
embedding_map = dict()
glove_file = 'glove.6B.50d.txt'
with open(glove_file) as f:
    for l in f.readlines():
        line = l.split()
        word = line[0]
        numbers = [float(n) for n in line[1:]]
        embedding_map[word] = numbers

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

print("Extracting Embeddings..")

embeddings_array = np.array(list(embedding_map.values()))
print ("PCA")
embeddings_3d_pca = PCA(n_components=3).fit_transform(embeddings_array)
print ("TSNE")
embeddings_3d_tsne = TSNE(n_components=3).fit_transform(embeddings_array)

print (embeddings_3d_pca.shape)
print (embeddings_3d_tsne.shape)




print ("pandas")
table = pd.DataFrame(data={'name':list(embedding_map.keys()),
                           'id': [i[1] for i in embedding_map.values()],
                           'x_pca': embeddings_3d_pca[:, 0],
                           'y_pca': embeddings_3d_pca[:, 1],
                           'z_pca': embeddings_3d_pca[:, 2],
                           'x_tsne': embeddings_3d_tsne[:, 0],
                           'y_tsne': embeddings_3d_tsne[:, 1],
                           'z_tsne': embeddings_3d_tsne[:, 2]
                           })

table.to_csv("knowledge_graph_3d_choords.csv")