---
layout: post
title:  "Molecular Generation using Junction Tree VAE using PyTorch"
author: meghana_kanapaneni
date: 2019-06-27
comments: True
mathjax: True
---
>Representation of Molecules can be done in the form of graphs. To existing generative models on graph data structures, we need better algorithms. Junction Tree VAE helps to address this issue and creates better molecular graphs.

<h2>Brief introduction</h2>
Molecular structure generation is one of the major parts of a material or drug discovery. This task involves continuous embedding and generation of molecular graphs.
Our junction tree variational autoencoder generates molecular graphs in two phases:<br>
(i)First, generating a tree-structured scaffold over chemical substructures<br>.
(ii)Combining them into a molecule with a graph message-passing network.

<h2>Synopsis</h2>
A molecular graph G is first decomposed into its junction tree T<sub>G</sub>, where each coloured node in the tree represents a substructure in the molecule.We then encode both the tree and graph into their latent embeddings z<sub>T</sub> and z<sub>G</sub>. To decode the molecule, we first reconstruct junction
tree from z<sub>T</sub> , and then assemble nodes in the tree back to the original molecule.
<center>{%include image.html url="\assets\img\jvae_1.png" %}</center>
<h2>Implementation</h2>
I'll be showing you how I built my Junction tree VAE in Pytorch. The dataset I used is ZINC dataset.The dataset contains smiles representation of molecules.I have also used RDKit to process the molecules.
<h3>I.Data Preprocessing</h3>
(i)Import the text file into our code.

```python
with open('train.txt') as f:
    data = [line.strip("\r\n ").split()[0] for line in f]
```

(ii)Convert each molecule to a Molecular tree. First, we have to decompose the molecule to a tree.

<h4>Tree Decomposition of Molecules:</h4>
A tree decomposition maps a graph G into a junction tree by contracting certain vertices into a single node so that G becomes cycle-free.
Formally, given a graph G, a junction tree TG = (V, E, X ) is a connected labeled tree whose node set is V = {C1, · · · , Cn} and edge set 
is E.Here X is vocabulary contains only cycles (rings) and single edges.

We first find simple cycles of given graph G, and its edges not belonging to any cycles.

Two simple rings are merged if they have more than two overlapping atoms. Each of those cycles or edges is considered as a 
cluster(clique).

Here cliques are nothing but the clusters. We will check the length of the clique and if it is more than 2, then we will check for the set of
intersection atoms in the neighborhood list of the cluster. If the intersection atom list is more than 2 we will merge them.

```python
    for i in range(len(cliques)):
        if len(cliques[i]) <= 2: continue
        for atom in cliques[i]:
            for j in nei_list[atom]:
                if i >= j or len(cliques[j]) <= 2: continue
                inter = set(cliques[i]) & set(cliques[j])
                if len(inter) > 2:
                    cliques[i].extend(cliques[j])
                    cliques[i] = list(set(cliques[i]))
                    cliques[j] = []
    
    cliques = [c for c in cliques if len(c) > 0]
    nei_list = [[] for i in range(n_atoms)]
    for i in range(len(cliques)):
        for atom in cliques[i]:
            nei_list[atom].append(i)
```

Next, a cluster graph is constructed by adding edges between all intersecting clusters. Finally, we select one of its spanning trees as the 
junction tree of G.

Here csr_matrix is creating a sparse matrix with a given number of rows and columns and minumum_spanning_tree is an inbuilt from scipy module to 
get the minimum spanning tree.

```python
    row,col,data = zip(*edges)
    n_clique = len(cliques)
    clique_graph = csr_matrix( (data,(row,col)), shape=(n_clique,n_clique) )
    junc_tree = minimum_spanning_tree(clique_graph)
```

Now, after collecting cliques and edges from tree decomposition, we construct a molecular tree using those cliques and edges.

<h3>II.Defining the model</h3>
Here we illustrate our model as JTNNVAE.

```python
class JTNNVAE(nn.Module):

    def __init__(self, vocab, hidden_size, latent_size, depthT, depthG):
        super(JTNNVAE, self).__init__()
        self.vocab = vocab
        self.hidden_size = int(hidden_size)
        self.latent_size = latent_size = latent_size / 2 #Tree and Mol has two vectors
        self.latent_size=int(self.latent_size)
        self.jtnn = JTNNEncoder(int(hidden_size),int(depthT), nn.Embedding(780,450))
        self.decoder = JTNNDecoder(vocab, int(hidden_size), int(latent_size), nn.Embedding(780,450))

        self.jtmpn = JTMPN(int(hidden_size), int(depthG))
        self.mpn = MPN(int(hidden_size), int(depthG))

        self.A_assm = nn.Linear(int(latent_size), int(hidden_size), bias=False)
        self.assm_loss = nn.CrossEntropyLoss(size_average=False)

        self.T_mean = nn.Linear(int(hidden_size), int(latent_size))
        self.T_var = nn.Linear(int(hidden_size), int(latent_size))
        self.G_mean = nn.Linear(int(hidden_size), int(latent_size))
        self.G_var = nn.Linear(int(hidden_size), int(latent_size))        
```
<h3>III.Training</h3>
As we have already seen in the synopsis,first we have to encode the graph and then tree and then decode the tree and then finally decode 
the graph.This will help us to make our work easy.
<h4>Graph Encoder:</h4>

(i) We encode the latent representation of G by a graph message-passing network<br>.
(ii)Each vertex v has a feature vector x<sub>v</sub> indicating the atom type, valence, and other properties. Similarly, each edge (u, v) ∈ E has a 
feature vector x<sub>uv</sub> indicating its bond type, and two hidden vectors ν<sub>uv</sub> and ν<sub>vu</sub> denoting the message from u to v and vice versa.

Here a1,a2 are two atoms having a bond. We get atom features from atom_features1. We also get bond features from bond_features1 function and we
concatenate them to an existing dimension of the atom.

```python  
            for atom in mol.GetAtoms():
                fatoms.append( atom_features1(atom) )
                in_bonds.append([])

            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                x = a1.GetIdx() + total_atoms
                y = a2.GetIdx() + total_atoms

                b = len(all_bonds) 
                all_bonds.append((x,y))
                fbonds.append( torch.cat([fatoms[x], bond_features1(bond)], 0) )
                in_bonds[y].append(b)

                b = len(all_bonds)
                all_bonds.append((y,x))
                fbonds.append( torch.cat([fatoms[y], bond_features1(bond)], 0) )
                in_bonds[x].append(b)
            
            scope.append((total_atoms,n_atoms))
            total_atoms += n_atoms
```

(iii)After T steps of iteration, we aggregate those messages as the latent vector of each vertex.

<h4>Tree Encoder:</h4>
(i) We similarly encode T<sub>G</sub> with a tree message passing network.

We construct a node graph and message graph.Node graph contains the information that's where the messages are connected to a index. The message
graph contain information that which message is in the invert direction.

```python
        for x,y in messages[1:]:
            mid1 = mess_dict[(x.idx,y.idx)]
            fmess[mid1] = x.idx 
            node_graph[y.idx].append(mid1)
            for z in y.neighbors:
                if z.idx == x.idx: continue
                mid2 = mess_dict[(y.idx,z.idx)]
                mess_graph[mid2].append(mid1)
```

(ii)In the first bottom-up phase, messages are initiated from the leaf nodes and propagated iteratively towards root.<br>
(iii)In the top-down phase, messages are propagated from the root to all the leaf nodes.

<h4>Tree Decoder:</h4>
We decode a junction tree T from its encoding z<sub>T</sub> with a tree structured decoder.Our tree decoder traverses the entire tree from the root,
and generates nodes in their depth-first order.<br>
<center>{%include image.html url="\assets\img\jvae_2.png" description="Tree Decoding Process" %}</center>

For every visited node, the decoder first makes a topological prediction whether this node has children to be generated. When a new child node is created, we predict its label and recurse this process. The decoder backtracks when a node has no more children to generate. Here we 
decode our tree and assemble the nodes in depth-first order.

<h4>Graph Decoder:</h4>
(i) The final step of our model is to reproduce a molecular graph G that underlies the predicted junction tree T.<br>
<center>{%include image.html url="\assets\img\jvae_3.png" %}</center>

(ii)We enumerate different combinations between red cluster C and its neighbors. Crossed arrows indicate combinations that lead to chemically infeasible molecules.<br>
(iii) Rank subgraphs at each node. The final graph is decoded by putting together all the predicted subgraphs.<br>

After training is done, save the model using torch.save().

<h3>IV.Generating sample molecules</h3>
This is how we sample new molecules. First, load the saved model and generate the new text using sample_prior() function.
```python
    model.load_state_dict(torch.load(path))
    torch.manual_seed(0)
    for i in range(10):
        print(model.sample_prior())
```
<h3>Results</h3>
Here are some samples I generated.
```python
    NCNCN
    O=C(CCNC(=O)c1ccccc1)Nc1cc(Cl)cc(Cl)c1
    c1ccccc1
    NC(N)N
    c1ccsc1
    CC(C)C
    C1CC1
    CC(C)C
    O=C=O
    CC(=O)C(C)C
```
Thanks for reading!!! Happy learning! :)


