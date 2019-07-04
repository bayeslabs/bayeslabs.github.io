
Before you dig into details of Recurrent Neural networks, if you are a Beginner I suggest you to read about RNN.

<b>Note</b>: To go through the article, you must have basic knowledge of neural networks and how Pytorch (a deep learning library) works. You can refer the some articles to understand these concepts:

In this post ,I am implementing a RNN model with Pytorch to generate SMILES.

Now in this we will learn 
<ul>
<li> Why/what are Recurrent Neural Networks?</li>
<li> Character level RNN model</li>
<li> RNN for Molecules (SMILES) Generation </li>
<li> Generating SMILES using RNN's</li>
</ul>
<b>Recurrent Neural Networks:</b> The idea behind RNN's is to make use of sequential information. RNN's are used for sequential data, like audio or sentences, where the order of the data plays an important role.
What makes Recurrent Networks so special? The core reason that recurrent nets are more exciting is that they allow us to operate over sequences of vectors:
{% include image.html url="/assets/img/1rnn.png" description="Unrolled RNN Architecture" %}

<b>Character-Level RNN Model:</b>  Okay, so we have an idea about what RNNs are, why they are super exciting, and how they work. We’ll now ground this in a fun application: We’ll train RNN character-level RNN models. That is, we’ll give the RNN a huge chunk of data(Smiles representation of molecules)and ask it to model the probability distribution of the next character in the sequence given a sequence of previous characters. This will then allow us to generate new smiles one character at a time.By the way, together with this post I am also releasing (code on Github…....link..................) that allows you to train char RNN model based on multi-layer LSTMs.
 <b>RNN for Molecules (SMILES) Generation-</b> In this Post, we want to show that recurrent neural networks can be trained as generative models for molecular structures, similar to statistical language models in natural language processing. We demonstrate that the properties of the generated molecules correlate very well with the properties of the molecules used to train the model.
  
  To connect chemistry with language, it is important to understand how molecules are represented. Usually, they are modeled by molecular graphs, also called Lewis structures in chemistry. In molecular graphs, atoms are labeled nodes. The edges are the bonds between atoms, which are labeled with the bond order (e.g., single, double, or triple).

However, in models for natural language processing, the input and output of the model are usually sequences of single letters, strings or words. We therefore employ the  <b>SMILES</b>  ( <b>S</b>  implified Molecular Input Line Entry System) format are the type of chemical notation that helps us to represent molecules and easy to used by the computers. It is a simple string representation of molecules, which encodes molecular graphs compactly as human-readable strings. SMILES is a formal grammar which describes molecules with an alphabet of characters, for example c and C for aromatic and aliphatic carbon atoms, O for oxygen, and −, =, and # for single, double, and triple bonds (see Figure 2).To indicate rings, a number is introduced at the two atoms where the ring is closed. For example, benzene in aromatic SMILES notation would be c1ccccc1.

{%include image.html url="\assets\img\smiles.png description="Examples of molecule and It's SMILES representation. To correctly create smiles, the model has to learn long-term dependencies, for example, to close rings (indicated by numbers) and brackets." %}
 <b>Generating SMILES using RNN's:</b>  I'll be showing you how I implemented my recurrent neural network in Pytorch. I trained it using the ChEMBL smiles Dataset ,which contains 2M smiles,and it is a manually curated database of bio-active drug-like molecules.
