---
layout: post
title:  "Objective Reinforced Genarative Adversarial Network(Part II)"
date: 2019-07-04
comments: True
mathjax: True
---
This blog is in continuation with the <a href = "https://haroon03.github.io/2019/07/03/Objective-Reinforced-Generative-Adversarial-Network-(Part-I).html">ORGAN Part I</a>, where I have thoroughly discussed about Generative Adversarial Network(GAN) and ORGAN. So, here I will walk you through the code of ORGAN to give you an overview.

<h3>ORGAN: MODEL</h3>

<h4><b>Dataset:</b></h4>

SMILES data set has been used for the training purpose of the GAN. 
<b>Simplified Molecular-Input Line-Entry System (SMILES)</b>. SMILES is a line notation for representing molecules and reactions.


{% include image.html align="center" url="\assets\img\SMILES.png" description="SMILES Representation"  %}

We load the Data-set and convert it into a string, Create a vocabulary of all the characters present in the data-set. The characters "\<bos>","\<eos>", "\<unk>" and "\<pad>" <b>(markers)</b> were added to the vocabulary.
     <li>"\<bos>": marks the beginning of sequence</li>
     <li>"\<eos>": marks end of sequences</li>
     <li>"\<unk>": specifies an unknown character</li>
     <li>"\<pad>": specifies padding</li>
Also the characters in vocabulary are indexed (c2i & i2c).We convert the smiles into tensors using the index.
Every time a smiles string is converted int tensor we add all the four markers at their specific locations. "bos" at the beginning of sequence, "eos" at the end of sequence, "unk" for characters unknown i.e. not in vocabulary. Padding is done to maintain a specific sequence length, 100 here.Now,we have the data that our computer can read and understand.

<b>Generator(G):</b>

It is a Recurrent Neural Network(RNN) with Long-short Term Memory (LSTM) cells. It is responsible for generating molecules that closely follows the distribution of training data. A generator can be assumed as a money forger. The Generator is initially trained on the training set to generate molecules.
LSTM layers are best for large sequrntial data. They have better memory retention power than GRU.
It takes the initial character(tensor) from sequence and predicts the next one until "\<eos>".It outputs the sequence(x), its length(lengths), and the current state.

     

```python
class Generator(nn.Module):
    def __init__(self, embedding_layer, hidden_size, num_layers, dropout):
        super(Generator, self).__init__()

        self.embedding_layer = embedding_layer
        self.lstm_layer = nn.LSTM(embedding_layer.embedding_dim,
                                  hidden_size, num_layers,
                                  batch_first=True, dropout=dropout)
        self.linear_layer = nn.Linear(hidden_size,
                                      embedding_layer.num_embeddings)

    def forward(self, x, lengths, states=None):
        x = self.embedding_layer(x)
        x = pack_padded_sequence(x, lengths, batch_first=True)
        x, states = self.lstm_layer(x, states)
        x, _ = pad_packed_sequence(x, batch_first=True)
        x = self.linear_layer(x)

        return x, lengths, states
```


What we are doing here is, we initially use the embedding layer to understand the relationship between the characters and generate a tensor with dimension equal to embedding dimension. Since the sequences are padded we pack the sequence using ```pack_padded_sequence```. What it does is it creates a tuple of two lists. One contains the data of all the sequences in the batch and the other holds the ```batch_size``` and tells us  how the elements are related to each other by the steps. 

You must be thinking why do padding and packing? Why not just simply pass the padded tensors to the RNN.

Well packing saves us a lot of computations. It does not perform computations on the padded sequences scince all the paddings are not included in the packed list of tensors thus resukts in saving a lot of time and energy. Just to give you an idea I will explain a small example:

```
Let's assume we have 6 sequences (of variable lengths) in total. You can also consider this number 6 as the batch_size hyperparameter. We have to pass these sequences through an RNN architecture(Assume LSTM), to do so we pad the sequence (with 0's) upto max length of a sequence.
 So the sequences are [1,2,3,4,5,6,7,8,9],[1,2,3,4,5,6,7,8],[1,2,3,4,5,6],[1,2,3,4],[1,2,3],[1,2].
 Note: The values may be different.
 
 So we have sequences of lengths 2,3,4,6,8,9. Now we pad them all to length upto 9.
 [1,2,3,4,5,6,7,8,9],
 [1,2,3,4,5,6,7,8,0],
 [1,2,3,4,5,6,0,0,0],
 [1,2,3,4,0,0,0,0,0],
 [1,2,3,0,0,0,0,0,0],
 [1,2,0,0,0,0,0,0,0].
 Lets say that we will matrix multiply the above padded_batch_of_sequences of shape (6, 9) with a weight matrix W of shape   (9, 3). We perfom 6x9=54 multiplication and 6x8=48 addition operations. Where most will be 0's.
 So during packed condition we only perform 32 multilications and 26 additions. Now considering this for thousands of sequences it will save us a lot of time and energy.

```

Pass this through the LSTM layer where it will understand the relation between the elemants and lern overtime the sequence patterns. Afterward , we have to convert the output from the LSTM into  padded batch output form (initial form).
That is done by ```pad_packed_sequence```.

<b>Discriminator(D):</b>

It plays the role of a cop who is trained to catch fake molecules generated by G.The Discriminator is composed of Convolutional Neural Networks(CNN), specifically designed for text classification. It gives a probability estimation of the molecule of either being fake(0/generated) or real(1/belongs to true/training data).
Initially we begin with embedding, that takes the character ids and generates a vector for each id. This is much more efficient than one hot encoding where each character is a vector containing zeros at all positions except one unique position which represents that chracter.

The Discriminator here uses a 2D- Convolution Layer.

```python
class Discriminator(nn.Module):
    def __init__(self, desc_embedding_layer, convs, dropout=0):
        super(Discriminator, self).__init__()

        self.embedding_layer = desc_embedding_layer
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(1, f, kernel_size=(
                n, self.embedding_layer.embedding_dim)
                       ) for f, n in convs])
        sum_filters = sum([f for f, _ in convs])
        self.highway_layer = nn.Linear(sum_filters, sum_filters)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.output_layer = nn.Linear(sum_filters, 1)

    def forward(self, x):
        x = self.embedding_layer(x)
        x = x.unsqueeze(1)
        convs = [F.elu(conv_layer(x)).squeeze(3)
                 for conv_layer in self.conv_layers]
        x = [F.max_pool1d(c, c.shape[2]).squeeze(2) for c in convs]
        x = torch.cat(x, dim=1)

        h = self.highway_layer(x)
        t = torch.sigmoid(h)
        x = t * F.elu(h) + (1 - t) * x
        x = self.dropout_layer(x)
        out = self.output_layer(x)

        return out
```



<h3>Training</h3> 
As discussed in previous post training has 2 phases.
Here we begin training the generator initially with training data.For each epoch the sequence is split into previous and next.  previous is sequence[:-1] and next is sequence[1:]. The generator predicts the next character and each time, loss is calculated and the generator is optimsed and parameters updated.
While training the Generator the discriminator is kept in eval mode(freezed).



```python
def _pretrain_generator_epoch(model, tqdm_data, criterion, optimizer):
    model.discriminator.eval()
    if optimizer is None:
        model.eval()
    else:
        model.train()
    postfix = {'loss': 0, 'running_loss': 0}
    for i, batch in enumerate(tqdm_data):
        (prevs, nexts, lens) = (data.to(device) for data in batch)
        outputs, _, _, = model.generator_forward(prevs, lens)
        loss = criterion(outputs.view(-1, outputs.shape[-1]),nexts.view(-1))
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        postfix['loss'] = loss.item()
        postfix['running_loss'] += (loss.item() - postfix['running_loss']) / (i + 1)
        tqdm_data.set_postfix(postfix)

    postfix['mode'] = ('Pretrain: eval generator' if optimizer is None else 'Pretrain: train generator')
                      
    return postfix
```      




Now we freeze the generator and train the discriminator. The training process of discriminator comprises of 2 parts. It is trained first on the training data with labels. Since the Data set has valid molecules all are labelled 1.
Loss is calculated for each prediction.
 
Simultaneously it is also trained on fake data generated by the generator, all labelled 0. 
Loss is calculated, generator is optimzed and all parameters updated.




```python
def _pretrain_discriminator_epoch(model, tqdm_data,
                                  criterion, optimizer=None):
    model.eval()
    if optimizer is None:
        model.eval()
    else:
        model.train()

    postfix = {'loss': 0,'running_loss': 0}
    for i, inputs_from_data in enumerate(tqdm_data):
        inputs_from_data = inputs_from_data.to(device)
        inputs_from_model, _ = model.sample_tensor(n_batch, 100)

        targets = torch.zeros(n_batch, 1, device=device)
        outputs = model.discriminator_forward(inputs_from_model)
        loss = criterion(outputs, targets) / 2

        targets = torch.ones(inputs_from_data.shape[0], 1, device=device)
        outputs = model.discriminator_forward(inputs_from_data)
        loss += criterion(outputs, targets) / 2

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        postfix['loss'] = loss.item()
        postfix['running_loss'] += (loss.item() - postfix['running_loss']) / (i + 1)                                    
        tqdm_data.set_postfix(postfix)

    postfix['mode'] = ('Pretrain: eval discriminator'
                       if optimizer is None
                       else 'Pretrain: train discriminator')
    return postfix
```


Now as we are done with pretraining both generator and the discriminator we begin with our reinforced learning with policy gradient method. We start with our smiles, converting them into molecules. Check if they are valid or invalid molecules. Remove the invalid smiles and map the valid molecules to their corresponding smiles. 

Using these the refrence smiles and reference molecules we initate our policy gradient method. Here the criterion for calculating loss of the generator is via policy gradient function and for discriminator Binary Cross Entropy with Logits loss(nn.BCEWithLogitsLoss()) which is sigmoid and BCEloss combined together. Both generator and discriminator use Adam optimizer to optimize its parameters.

Now we again train the generator, but this time in a different manner. While pretraining generator we calculated the loss based on the difference between predicted token and the actual token that was supposed to be predicted. But here we first get the rewards from the reward metrics. How do we do that?
Pretty simlple, create an empty tensor (prev) insert the index of "\<bos>" token and feed it to the generator. The token initialises the prediction of the sequence. We calculate the reward, using the reward metrics, for each predicted token. The generated sequence and rewards are forwarded to the policy gradient function to calculate the loss.
   
Policy Gradient Function to calculate Loss.



```python 
     class PolicyGradientLoss(nn.Module):
    def forward(self, outputs, targets, rewards, lengths):
        log_probs = F.log_softmax(outputs, dim=2)
        items = torch.gather(
            log_probs, 2, targets.unsqueeze(2)
        ) * rewards.unsqueeze(2)
        loss = -sum(
            [t[:l].sum() for t, l in zip(items, lengths)]
        ) / lengths.sum().float()
        return loss
```


Now, we have the loss for the generated molecule we optimize the generator, backpropogate and calculate the gradients but bfore updating we clip the gradients so that they remain under a cetain threshold. Finally we update the generator gradients.



```python
    def _policy_gradient_iter(self, model, train_loader,
                              criterion, optimizer, iter_):
        smooth = self.config.pg_smooth_const if iter_ > 0 else 1
        gen_postfix = {'generator_loss': 0,'smoothed_reward': 0}
        gen_tqdm = tqdm(range(self.config.generator_updates),
                        desc='PG generator training (iter #{})'.format(iter_))
        for _ in gen_tqdm:
            model.eval()
            sequences, rewards, lengths = model.rollout(
                self.config.n_batch, self.config.rollouts,
                self.ref_smiles, self.ref_mols, self.config.max_length
            )
            model.train()
            lengths, indices = torch.sort(lengths, descending=True)
            sequences = sequences[indices, ...]
            rewards = rewards[indices, ...]

            generator_outputs, lengths, _ = model.generator_forward(sequences[:, :-1], lengths - 1)
            generator_loss = criterion['generator'](
            generator_outputs, sequences[:, 1:], rewards, lengths)
            optimizer['generator'].zero_grad()
            generator_loss.backward()
            nn.utils.clip_grad_value_(model.generator.parameters(), self.config.clip_grad)
            optimizer['generator'].step()
            gen_postfix['generator_loss'] += (generator_loss.item() - gen_postfix['generator_loss'] ) * smooth
            mean_episode_reward = torch.cat(
                [t[:l] for t, l in zip(rewards, lengths)]
            ).mean().item()
            gen_postfix['smoothed_reward'] += (mean_episode_reward - gen_postfix['smoothed_reward']) * smooth
            gen_tqdm.set_postfix(gen_postfix)
```



We have successfully updated and trained the generator, now its time to train the discriminator.
We generate samples from the generator in batches. We iterate over all the batches and through each molecule to predict the probability of it being fake. Again determine the loss and update the gradients. This is carried out for few epochs (10 in this case).



 ```python
 discrim_postfix = {'discrim-r_loss': 0}
        discrim_tqdm = tqdm(
            range(self.config.discriminator_updates),
            desc='PG discrim-r training (iter #{})'.format(iter_) )
        for _ in discrim_tqdm:
            model.generator.eval()
            n_batches = (len(train_loader) + self.config.n_batch - 1 ) // self.config.n_batch
            sampled_batches = [model.sample_tensor(self.config.n_batch, self.config.max_length)[0] for _ in range(n_batches)]   
            for _ in range(self.config.discriminator_epochs):
                random.shuffle(sampled_batches)

                for inputs_from_model, inputs_from_data in zip(sampled_batches, train_loader):

                    inputs_from_data = inputs_from_data.to(model.device)

                    discrim_outputs = model.discriminator_forward(inputs_from_model)
                    discrim_targets = torch.zeros(len(discrim_outputs),1, device=model.device) 
                    discrim_loss = criterion['discriminator'](discrim_outputs, discrim_targets ) / 2                   

                    discrim_outputs = model.discriminator_forward(inputs_from_data)
                    discrim_targets = torch.ones(len(discrim_outputs), 1, device=model.device)
                    discrim_loss += criterion['discriminator'](discrim_outputs, discrim_targets) / 2
                    optimizer['discriminator'].zero_grad()
                    discrim_loss.backward()
                    optimizer['discriminator'].step()

                    discrim_postfix['discrim-r_loss'] += (discrim_loss.item() - discrim_postfix['discrim-r_loss']) * smooth                                                               
            discrim_tqdm.set_postfix(discrim_postfix)

        postfix = {**gen_postfix, **discrim_postfix}
        postfix['mode'] = 'Policy Gradient (iter #{})'.format(iter_)
        return postfix
```

We have trained our GAN completely. Here the role of reinforcement learning boosts the accuracy and predictibility of our model. Policy Gradient Function guides the generator to generate molecules with certain properties that are defined in the reward metrics.
For breif expalinations of the reward metrics used in this model here are some links.


<a href = "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3524573/"> Quantitative Estimation of Drug-Likeness</a> 


<a href = "https://arxiv.org/abs/1803.09518">Frēchet ChemNet Distance</a>


<a href = "https://en.wikipedia.org/wiki/Partition_coefficient">Partition Coefficient</a> 


<a href = "https://jcheminf.biomedcentral.com/articles/10.1186/s13321-015-0069-3">Tanimoto Similarity </a>


<a href="https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3225829/">Sythectic Accessibility Score</a>


<a href = "https://peter-ertl.com/reprints/Ertl-JCIM-48-68-2008.pdf">Natural Product Likeness Score</a>

You can find the complete code at <a href=" github link ">github</a>.
