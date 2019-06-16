from process_data import DecagonData
import matplotlib.pyplot as plt
import numpy as np
import pickle

NUM_EDGE = 1317
et = [i for i in range(NUM_EDGE)] + [i for i in range(NUM_EDGE)]         # ordered edge types
data = DecagonData(et)
data = data.adj_mats_orig[1, 1]

# ########################### For Embedding Check ########################## #
adj = data.adj_mats_orig
adj[1, 1][0].diagonal().max()


# ########################### Histogram of DD Edge Type ########################## #
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


# ########################### Sample training DD edge types ########################## #
tmp = np.array([data[i].nnz for i in range(NUM_EDGE)])

lower_bound = 500
higher_bound = 1500

boolean = np.logical_and(tmp > lower_bound, tmp <= higher_bound)
indices = np.nonzero(boolean)[0].tolist()

with open("./data_decagon/training_samples.pkl", "wb") as f:
    pickle.dump(indices, f)


