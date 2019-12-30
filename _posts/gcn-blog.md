Graph Convolutional Network
---------------------------

### A Brief History

History of develoment : Refer to 1

### Introduction to GCN

What is Graph Convolutional Network (GCN) ?

We observe many graph-structured real-world scenarios such as social
media, other problems, Protein-Protein Interaction etc. But, often times
it's a challenging task to solve learning problems on graphs due to
their high complexity.

Graph Convolutional Network is a type of neural network that operates on
graphs. Main goal is to embed a neural network into the graph structure
itself which involves storing states for each node and using an
adjacency matrix to propagate those states to the nodes’ neighbors.
States for each node could be their features (given or derived using
random walk.) Given a graph G = (V, E), GCN takes following input :

​1. An input feature matrix N × F⁰ feature matrix, X, where N is the
number of nodes and F⁰ is the number of input features for each node,
and \
2. An N × N matrix representation of the graph structure i.e. adjacency
matrix A.

And produces an F(L)-dimensional output for each node Z i.e. an N×F(L)
output feature matrix, L is number of layers.

Each GCN layer can be represented as a function of previous GCN layer
and adjacency matrix of the graph:\
 H(l+1) = f(H(l) , A), \
where H(l) is feature matrix of shape N x F(l) for layer l [F(l) is
number of features in layer l]. We define H(0) = X. \
And f is a non-linear activation function such as ReLU. \
 \
Each row of feature matrix corresponds to feature representation of a
node.

\

### Propagation Rule

Now that we know what inputs and output of a GCN look like, let's see a
simple propagation rule that can be defined as: \
 f(H(l), A) = f(A H(l) W(l)) [Tkipf], \
 where W(l) is a weight matrix for the l-th GCN layer, and f is a
non-linear activation function like ReLU. \
The weight matrix has dimensions Fⁱ × Fⁱ⁺¹; in other words the size of
the second dimension of the weight matrix determines the number of
features at the next layer. [Elaborate]

Now, let's understand what is this propagation rule doing: \
 when we multiply A with H(l), a nodes' features in layer l are updated
as the sum of their neighbors' features. [EXAMPLE] \
 and when we multiply above with weights and apply non-linear activation
to it, the feautre representations are transformed. \
Order of matrix muliplication doesn't matter, since it's associative
i.e. [ (AB)C = A(BC) ]

Let's call the first multiplication 'aggregation' (since neighbors'
features are being summed) and second 'transformation' as described in
this blog.

You may have already noticed, there are few problems with the
propagation rule defined above : \
1. In aggregation step, a node's own feature is not being included. To
fix this, we add self-loop to each node. [EXAMPLE] \
2. Second problem is that, typically A is not normalized, i.e. all rows
don't sum up to one, therefore when we calculate feature representation
of each node, a node with higher degree will have higher feature value
as compared to the node which has lower degree. [Eigenvalue
Tkipf][EXAMPLE] \
This can cause vanishing or exploding gradients, and could also cause
problem for gradient descent based algorithms because of scale
difference in features. \
To fix this problem, we normalize A as described by [Tkipf] such that
all rows sum upto one : \
 A\_h = D\*\*-1 \* A, \
 D is diagonal node degree matrix of A, and A\_h is the normalized A.

Now, when we multiply features with normalized A i.e. A\_h, we get the
mean of features of neighbor nodes, as opposed to their direct sum as in
the case of un-normalized A.

So far, we have seen two types of propagation rule : \
 1. f(Hⁱ, A) = σ(AHⁱWⁱ), \
 2. f(Hⁱ, A) = σ(D⁻¹ÂHⁱWⁱ), where Â = A + I, I is the identity matrix,
and D⁻¹ is the inverse degree matrix of Â.

Now, let's discuss spectral propagation rule, which was proposed in the
paper "GCN for semi-supervised Node Classification" by Tkipf and
Welling. \
 f(H(l), A) = sigma(D\^−1/2 \* A\_h \* D\^−1/2 H(l) W(l) ) \
 Difference between this propagation rule and the one described above is
in the way they normalize A. \
 Now when we multiply A\_h with features, we observe something very
interesting. When computing the aggregate feature representation of the
ith node, we not only take into consideration the degree of the ith
node, but also the degree of the jth node. Similar to the mean rule, the
spectral rule normalizes the aggregate s.t. the aggregate feature
representation remains roughly on the same scale as the input features.
However, the spectral rule weighs neighbor in the weighted sum higher if
they have a low-degree and lower if they have a high-degree. This may be
useful when low-degree neighbors provide more useful information than
high-degree neighbors.[Tobias][Read this section of paper.]

#### Summarize all three propagetion rules

How we normalize A, and how it affects the way neighboring nodes are
aggregated.

\

### Semi-supervised learning

\
\
\
\

#### Sources :

[1] http://tkipf.github.io/graph-convolutional-networks/

[2] https://link.springer.com/article/10.1186/s40649-019-0069-y

[3]
http://web.cs.ucla.edu/\~patricia.xiao/files/Reading\_Group\_20181204.pdf

[4]
http://snap.stanford.edu/proj/embeddings-www/files/nrltutorial-part2-gnns.pdf

[5]
https://missinglink.ai/guides/convolutional-neural-networks/graph-convolutional-networks/
