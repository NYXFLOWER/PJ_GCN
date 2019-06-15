from process_data import DecagonData
import matplotlib.pyplot as plt
import numpy as np

NUM_EDGE = 1317
et = [i for i in range(NUM_EDGE)] + [i for i in range(NUM_EDGE)]         # ordered edge types
data = DecagonData(et)


# ########################### For Embedding Check ########################## #
adj = data.adj_mats_orig
adj[1, 1][0].diagonal().max()


# ########################### Histogram of DD Edge Type ########################## #
data = data.adj_mats_orig[1, 1]
tmp = [data[i].nnz for i in range(NUM_EDGE)]
tmp = np.sort(tmp)

num = plt.hist(tmp, bins=57)
plt.xlabel('Number of Times A D-D Edge Type Occurs')
plt.ylabel('Numbers of D-D Edge Type')
plt.title('Frequency Distribution Histogram of D-D Edge Type')
plt.grid()
# plt.yscale('log')
plt.savefig('hist_dd_edge.png')
plt.show()
