import csv
import numpy as np
import scipy.sparse as sp
import decagon.utility.preprocessing as preprocessing
import pickle


class DecagonData:
    def __init__(self, et):
        """:param num: load num+1 edge types in order"""
        # load data
        print("loading...")

        temp = ''
        with open(temp + 'data_decagon/graph_num_info.pkl', 'rb') as f:
            [num_gene, num_drug, num_edge_type, num_drug_additional_feature] = pickle.load(f)

        # gene-gene
        gene_adj = sp.load_npz(temp + "data_decagon/gene-sparse-adj.npz")
        print("load gene_gene finished!")

        # gene-drug
        gene_drug_adj = sp.load_npz(temp + "data_decagon/gene-drug-sparse-adj.npz")
        drug_gene_adj = sp.load_npz(temp + "data_decagon/drug-gene-sparse-adj.npz")
        print("load gene_drug finished!")

        # drug-drug
        drug_drug_adj_list = []
        for i in et:
            drug_drug_adj_list.append(sp.load_npz("".join([temp + "data_decagon/drug-sparse-adj/type_", str(i), ".npz"])))

        print("load drug_drug finished!")

        drug_feat_sparse = sp.load_npz(temp + "data_decagon/drug-feature-sparse.npz")
        print("load drug_feature finished!")

        # -------------------------- gene feature --------------------------
        # featureless (genes)
        gene_feat = sp.identity(num_gene)
        gene_nonzero_feat, gene_num_feat = gene_feat.shape
        gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

        # drug vectors with additional features (single side effect)
        drug_nonzero_feat, drug_num_feat = drug_feat_sparse.shape[1], np.count_nonzero(drug_feat_sparse.sum(axis=0))
        drug_feat = preprocessing.sparse_to_tuple(drug_feat_sparse.tocoo())

        # data representation
        self.adj_mats_orig = {
            (0, 0): [gene_adj, gene_adj.transpose(copy=True)],
            (0, 1): [gene_drug_adj],
            (1, 0): [drug_gene_adj],
            (1, 1): drug_drug_adj_list + [x.transpose(copy=True) for x in drug_drug_adj_list],
        }

        gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()
        drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]
        self.degrees = {
            0: [gene_degrees, gene_degrees],
            1: drug_degrees_list + drug_degrees_list,
        }

        # data representation
        self.num_feat = {
            0: gene_num_feat,
            1: drug_num_feat,
        }
        self.num_nonzero_feat = {
            0: gene_nonzero_feat,
            1: drug_nonzero_feat,
        }
        self.feat = {
            0: gene_feat,
            1: drug_feat,
        }

        self.edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in self.adj_mats_orig.items()}
        self.edge_type2decoder = {
            (0, 0): 'bilinear',
            (0, 1): 'bilinear',
            (1, 0): 'bilinear',
            (1, 1): 'dedicom',
        }

        self.edge_types = {k: len(v) for k, v in self.adj_mats_orig.items()}
        self.num_edge_types = sum(self.edge_types.values())
        print("Edge types:", "%d" % self.num_edge_types)
        print("======================================================")


    def build_original(self):
        pp_f = "data_decagon/PP-Decagon_ppi.csv"
        dd_f = "data_decagon/bio-decagon-combo.csv"
        dp_f = "data_decagon/bio-decagon-targets.csv"
        ds_f = "data_decagon/bio-decagon-mono.csv"
        p_set, d_set, combo_set, mono_set = set(), set(), set(), set()
        pp_list, ddt_list, dp_list, ds_list = [], [], [], []

        a, b, c = 0, 0, 0  # temp variables

        # 1. Protein-Protein Association Network
        with open(pp_f, 'r') as f:
            ppi = csv.reader(f)
            next(ppi)
            for [g1, g2] in ppi:
                a, b = int(g1), int(g2)
                p_set.add(a)
                p_set.add(b)
                pp_list.append((a, b))
        # 2. Drug-Drug Association Network
        with open(dd_f, "r") as f:
            ppi = csv.reader(f)
            next(ppi)
            for [d1, d2, t, n] in ppi:
                a, b, c = int(t.split('C')[-1]), int(d1.split('D')[-1]), int(d2.split('D')[-1])
                combo_set.add(a)
                d_set.add(b)
                d_set.add(c)
                ddt_list.append((b, c, a))
        # 3. Drug-Protein Association Network
        with open(dp_f, "r") as f:
            ppi = csv.reader(f)
            next(ppi)
            for [d, p] in ppi:
                a, b = int(d.split('D')[-1]), int(p)
                d_set.add(a)
                p_set.add(b)
                dp_list.append((a, b))
        # 4. Drug-SideEffect Association Network
        with open(ds_f, "r") as f:
            ppi = csv.reader(f)
            next(ppi)
            for [d, e, n] in ppi:
                a, b = int(e.split('C')[-1]), int(d.split('D')[-1])
                mono_set.add(a)
                d_set.add(b)
                ds_list.append((b, a))

        num_gene = p_set.__len__()
        num_drug = d_set.__len__()
        num_edge_type = combo_set.__len__()
        num_drug_additional_feature = mono_set.__len__()

        # -------------------------- gene adj --------------------------
        gene_to_old = list(p_set)
        gene_to_new = sp.csr_matrix((range(num_gene), ([0] * num_gene, gene_to_old)))

        drug_to_old = list(d_set)
        drug_to_new = sp.csr_matrix((range(num_drug), ([0] * num_drug, drug_to_old)))

        edge_type_to_old = list(combo_set)
        edge_type_to_new = sp.csr_matrix((range(num_edge_type), ([0] * num_edge_type, edge_type_to_old)))

        side_effect_to_old = list(mono_set)
        side_effect_to_new = sp.csr_matrix(
            (range(num_drug_additional_feature), ([0] * num_drug_additional_feature, side_effect_to_old)))

        r, c = [], []
        array_length = len(pp_list)
        # -------------------------- gene-gene adj --------------------------
        for i in range(array_length):
            r.append(gene_to_new[0, pp_list[i][0]])
            c.append(gene_to_new[0, pp_list[i][1]])
        gene_adj = sp.csr_matrix(([1] * array_length, (r, c)), shape=(num_gene, num_gene))
        gene_degrees = np.array(gene_adj.sum(axis=0)).squeeze()

        r, c = [], []
        array_length = len(dp_list)
        # -------------------------- drug(row)-gene(col) adj --------------------------
        for i in range(array_length):
            r.append(drug_to_new[0, dp_list[i][0]])
            c.append(gene_to_new[0, dp_list[i][1]])
        drug_gene_adj = sp.csr_matrix(([1] * array_length, (r, c)), shape=(num_drug, num_gene))
        gene_drug_adj = drug_gene_adj.transpose(copy=True)

        r = {}
        array_length = len(ddt_list)
        # -------------------------- drug-drug adj list --------------------------
        for i in range(array_length):
            c = edge_type_to_new[0, ddt_list[i][2]]
            if c not in r:
                r[c] = [drug_to_new[0, ddt_list[i][0]]], [drug_to_new[0, ddt_list[i][1]]]
            else:
                r[c][0].append(drug_to_new[0, ddt_list[i][0]])
                r[c][1].append(drug_to_new[0, ddt_list[i][1]])
        drug_drug_adj_list = []
        for i in range(num_edge_type):
            drug_drug_adj_list.append(
                sp.csr_matrix(([1] * len(r[i][0]), (r[i][0], r[i][1])), shape=(num_drug, num_drug)))
        drug_degrees_list = [np.array(drug_adj.sum(axis=0)).squeeze() for drug_adj in drug_drug_adj_list]

        # -------------------------- gene feature --------------------------
        # featureless (genes)
        gene_feat = sp.identity(num_gene)
        gene_nonzero_feat, gene_num_feat = gene_feat.shape
        gene_feat = preprocessing.sparse_to_tuple(gene_feat.tocoo())

        # drug vectors with additional features (single side effect)
        r, c = list(range(num_drug)), list(range(num_drug))
        for (a, b) in ds_list:
            r.append(drug_to_new[0, a])
            c.append(side_effect_to_new[0, b] + num_drug)
        array_length = num_drug + len(ds_list)
        drug_feat = sp.csr_matrix(([1] * array_length, (r, c)),
                                  shape=(num_drug, num_drug + num_drug_additional_feature))

        drug_nonzero_feat, drug_num_feat = drug_feat.shape[1], np.count_nonzero(drug_feat.sum(axis=0))
        drug_feat = preprocessing.sparse_to_tuple(drug_feat.tocoo())

        # data representation
        self.adj_mats_orig = {
            (0, 0): [gene_adj, gene_adj.transpose(copy=True)],
            (0, 1): [gene_drug_adj],
            (1, 0): [drug_gene_adj],
            (1, 1): drug_drug_adj_list + [x.transpose(copy=True) for x in drug_drug_adj_list],
        }
        self.degrees = {
            0: [gene_degrees, gene_degrees],
            1: drug_degrees_list + drug_degrees_list,
        }

        # data representation
        self.num_feat = {
            0: gene_num_feat,
            1: drug_num_feat,
        }
        self.num_nonzero_feat = {
            0: gene_nonzero_feat,
            1: drug_nonzero_feat,
        }
        self.feat = {
            0: gene_feat,
            1: drug_feat,
        }

        self.edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in self.adj_mats_orig.items()}
        self.edge_type2decoder = {
            (0, 0): 'bilinear',
            (0, 1): 'bilinear',
            (1, 0): 'bilinear',
            (1, 1): 'dedicom',
        }

        self.edge_types = {k: len(v) for k, v in self.adj_mats_orig.items()}
        self.num_edge_types = sum(self.edge_types.values())
        print("Edge types:", "%d" % self.num_edge_types)
