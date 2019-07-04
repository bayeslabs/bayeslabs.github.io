---
layout: post
title:  "Non Coding Variant Effect Prediction using Basenji"
date: 2019-07-02
comments: True
mathjax: True
---
<h2>Introduction</h2>
Human DNA is about 99.5% identical from person to person. However, there are small differences that make each person unique. These differences are called variants. And these variants are linked to certain health conditions.
Variants are of two types, Coding and Non-coding. The DNA which encodes the proteins are called coding DNA whereas the DNA which do not encode proteins are called Non-coding DNA.

<h2>Why Non Coding Variants?</h2>
<br>Only about 1 percent of DNA is made up of protein-coding genes; the other 99 percent is noncoding. Scientists once thought noncoding DNA was “junk,” with no known purpose. However, it is becoming clear that at least some of it is integral to the function of cells, particularly the control of gene activity and majority of the diseases are caused by these non-coding variants
However, non-coding genome is larger than the protein coding counterpart and contains structural, regulatory , transcribed information that needs to be incorporated into genome annotations to use genomic information in healthcare.

<h2>Predicting the effects of non-coding variants</h2>
In order to understand the genetic basis of disease, we need to understand the impact of variant across the genome. In this way, we can find the cure for most of the diseases. 
So, We need to build models that can predict molecular phenotypes directly from biological sequences to probe the association between the  phenotype and the genetic variation.
Thus, sequence based deep learning models can be used for assessing the impact of such variants which offer a promising approach to find potential drivers of complex phenotypes.
In this, we will learn about one such sequence based deep learning model, Basenji.

<h2>Basenji</h2>
<h3>Overview</h3>
Models for predicting phenotypic outcomes from genotypes have important applications to understanding genomic function and improving human health. Basenji is a machine-learning system to predict cell-type–specific epigenetic and transcriptional profiles in large mammalian genomes from DNA sequence alone.
Numerous lines of evidence suggest that many noncoding variants influence traits by changing gene expression. Thus, gene expression offers a tractable intermediate phenotype for which improved modeling would have great value

<h3>Model Architecture</h3>
Basenji is basically a CNN with three layers in its architecture I,e, convolution layers, dilated convolution layers and a final prediction layer.

<h3>Input</h3>
The model accepts much larger (2^17=) 131-kb regions as input.

<h3>Architecture</h3>
It performs multiple layers of convolution and pooling to transform the DNA to a sequence of vectors representing 128-bp regions.To share information across long distances, we then apply several layers of densely connected dilated convolutions. After these layers, each 128-bp region aligns to a vector that considers the relevant regulatory elements across a large span of sequence. Finally, we apply a final width-one convolutional layer to parameterize a multitask Poisson regression on normalized counts of aligned reads to that region for every data set provided . That is, the model's ultimate goal is to predict read coverage in 128-bp bins across long chromosome sequences.
We used a Basenji architecture with four standard convolution layers, pooling in between layers by two, four, four, and four to a multiplicative total of 128; seven dilated convolution layers; and a final convolution layer to predict the 4229 coverage data sets. We optimized all additional hyperparameters using Bayesian optimization parameters.

<h3>Output:</h3>
It has a final convolution layer to predict the 4229 coverage datasets.

<h3>Distal regulatory elements</h3>
Dilated convolutions extend the reach of our model to view distal regulatory elements at distances beyond previous models. We devised a method to quantify how distal sequence influences a Basenji model's predictions and applied it to produce saliency maps for gene regions. 
 <br> /textbf{Saliency score= ∑ 128-bp bin representations. Gradient of the model predictions}
<br>Peaks in this saliency score detect distal regulatory elements, and its sign indicates enhancing (+) versus silencing (−) influence.

And the output from the dilated convolution layers is fed into the prediction layer which predicts the 4229 coverage datasets. Coverage tells the number of unique reads that include a given nucleotide in the reconstructed sequence.
Given a SNP–gene pair, we define its SNP expression difference (SED) score as the difference between the predicted CAGE coverage at that gene's TSS (or summed across multiple alternative TSS) for the two alleles.
                          \textbf{SED= | Allele1 – Allele2|}
<br>By considering this |SED-LD| score, all the variants are ranked. And this gives the information about the causal variants. And then, these variants can be linked to disease loci and we can thus know the genetic basis of the disease.


