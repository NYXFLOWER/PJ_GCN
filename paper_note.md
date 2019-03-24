### Modeling Polypharmacy Side Effects with Graph Convolutional Network

Home Page: http://snap.stanford.edu/decagon/

#### 1. Dataset

The original dataset is a huge multimodal graph.

__Node:__
Each node either represents a drug(D) or a protein(P).

__Edge with label:__
Edges are undirected and labeled: $E=\{(\text{node1, label, node2})\}$. There are three types of nodes:
1) D-P link $(n_d, t, n_p)$: drug $d$ targets on protein $p$
2) P-P link $(n_{p1}, b, n_{p2})$: protein $p_1$ and protein $p_2$ have physical binding
3) D-D link $(n_{d1}, r_i, n_{d2})$: drug $d_1$ and drug $d_2$ can cause multipharmacy side effect $r_i$

This means all D-P / P-P links have the same label $t$ / $b$, but the label of a D-D link is chosen from $\{r_i\}_{i=1,...,N}$, where each $r_i$ represents a side effect. Note that between a pair of drugs there might be more than one links with different labels (a pair of drug might cause more than one side effects).

Use $L$ represents all kinds of labels.

#### 2. Method

__Data Drive:__

80% - training, 10% - validation, 10% - testing of drug-drug edges

__GCN Encoder:__

A network is designed to embed nodes. For node $n_i$, the input is its feature vector $\mathbf{x_i}$ and the output is a $d$-dimensional embeding vector $\mathbf{z_i}$. 

Layer: 
There are $K$ layers in this embedding network, where $K$ is less than the maximum number of neighborhood order over the graph. Using $h_i^k$ to denote the hidden state of node $n_i$ in the $k^{th}$ layer. $\mathcal{N}_l^i$ denoting the set of neighbors of node $n_i$ under relation $l$.
The input to the first layer are node feature vectors, $h_i^{(0)} = \mathbf{x}_i$. The output of the $K^{th}$ layer are the final embedding for each node, $\mathbf{z}_i = h_i^{K}$
Function for the $k^{th}$ layer :
$$
h_i^{k+1} = \phi(\sum_r \sum_{j\in \mathcal{N}_r^i} c_r^{ij} \mathbf{W}_r^{(k)} \mathbf{h}_j^{(k)} + c_r^i \mathbf{h}_i^{(k)}) \tag{1}
$$
, where $\phi$ is an non-linear element-wise activation function (i.e., ReLU), and 
$$
c_r^{ij} = \cfrac{1}{\sqrt{|\mathcal{N}_r^i| |\mathcal{N}_r^j|}}\ ,\qquad \qquad c_r^i = \cfrac{1}{\sqrt{|\mathcal{N}_r^i|}}
$$

__Tensor Factorization Decoder__ (DEDICOM):

The probability of the existence of the edge $(n_i, l, n_j)$ is computed by two functions in sequence, for any $l \in L$.

Tensor factorization:
$$
g(n_i, l, n_j)=\begin {cases}
\mathbf{z}_i^\top \mathbf{D}_l \mathbf{R} \mathbf{D}_l \mathbf{z}_j & z_i\ and\ z_j\ are\ drugs \\
\mathbf{z}_i^\top \mathbf{M}_l \mathbf{z}_j & otherwise
\end {cases} \tag{2}
$$

Sigmoid function:
$$
p_l^{ij} = p((n_i, l, n_j) \in \mathcal{R}) = \sigma(g(\mathbf{z}_i, l, \mathbf{z}_j)) \tag{3}
$$
where $\mathcal{R}$ is the set of D-D edges.

__Loss Function__:

The cross-entropy loss for a single edge $(n_i, l, n_j)$:
$$
J_l(i,j) = −\log p_l^{ij} − \mathbb{E}_{n∼P_l(j)} \log(1−p_l^{in} )\tag{4}
$$
The final loss function is
$$
J = \sum_{(n_i, l, n_j)\in \mathcal{R}} J_l(i, j) \tag{5}
$$
__Optimization__:

Parameters: $\mathbf{D}_l,\ \mathbf{R},\ \mathbf{M}_l,\ \mathbf{W}_l$

Optimizer: Adam(learning_rate=0.001, max_epoch=100, early_stopping=2)

Initialization: Xavier (Glorot and Bengio, 2010) accordingly normalize node feature vectors.

Dropout: regular dropout (Srivastava et al., 2014) to hidden layer units (Eq(1)).

Gradient Descent: mini-batching - sampling a fixed number of contributions to the loss function in Eq(5).

#### 3. Need to read Code...

1. How to construct node feature vectors?
2. In Eq(1), when  the neighbour of node is a protein, if there is a bias?