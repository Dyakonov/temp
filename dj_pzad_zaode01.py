import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix

data = pd.read_csv('train.csv')

ss = csr_matrix((data['sum'].values, (data.id.values - 1, data.date.values - 1)))

a = []
for i in range(110000):
    sh = ss[i,:].toarray().ravel()
    h = sh[4:].reshape(-1, 7)
    g = (((h>0).cumsum(axis=1) == 1) * h).sum(axis=1)
    
    j = np.argmax(np.dot (np.arange(62), csr_matrix((np.ones(62), (np.arange(62), g)), shape=(62, 17)).toarray()))
    a.append(j)

pd.DataFrame({'id': np.arange(1, 110001), 'sum':a}).to_csv('dummy_benchmark.csv', index=False)
