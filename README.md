
# Pynet
#### Simple neural network toolbox specifically designed to train networks simultaneously for evolution optimization.

# Installation
Installing to use in your own scripts in a virtual environment?

`pip install git+https://github.com/pmp47/Pynet`

Installing to edit this code and contribute? Clone/download this repo and...

`pip install -r requirements.txt`


Want to use a GPU? It's recommended to use a more robust package installer...

`conda install tensorflow-gpu`


# Requirements/Features

Pynets have been intentially designed to have some unique features:
* Easily implimented on CPU or GPU.
* Able to train multiple networks simultaneously.
* Modular, so new developments can be easily tested and applied such as new transfer functions.
* Abstract optimization so more focus is on quality of data, rather than exact form of the network.

# Motivation
The main motivation for developing this toolbox was to demonstrate how simple neural networks are to utilize and impliment. Other toolboxes out there are more complicated and less modular.
* Tensorflow is quite popular and provides a great foundation for achieving the first requirement.
* Pynets are groups of identically structured networks which satisfies the second requirement.
* The class division of the categories of Pynets and the wrapping around Tensorflow satisifes the modularity requirement.
* There was no optimization toolbox out there to apply to hyper-parameters like those of an entire neural network so the evolution script provides this ability.

# Usage
##### The simplest way to use a Pynet:
```python
from pynet import Pynet, Tests

#load test classification data such as the classic iris set
X,T = Tests.Utils.import_classification_data('iris')

#set to True to use GPU hardware, usually orders of magnitude faster than CPU
useGPU = False

#number of copies of the neural network in the group -> affects vram reqs and speed
groupSize = 8

#create a simple Pynet group of a Patternnet neural network
nnet = Pynet.Models.PatternnetGroup(groupSize,useGPU)

#configure this Pynet's graph
nnet.ConfigureGraph()

#train the pynet on the input/target datasets
nnet.Train(X,T)

#simulate the output of the network given input data
Y = nnet.Sim(X)

```
##### Evolution Optimization:
Don't know which model to use? Or none of them really get the job done? A pynet may have its structure optimized through an evolutary process. This means you only need to control computing resources rather than the neural network structure.
```python
#capcity of evolving population -> higher takes longer to progress through generations
pop_cap = 25

#method used to evaluate the fitness of each pynet
fitness_method = 'acc'

#create an evolution environment and produce an evolved Pynet
evolved_pynet = Pynet.Evolve(X,T,useGPU,pop_cap,fitness_method,time_limit_minutes=60)

#configure this Pynet's graph
evolved_pynet.ConfigureGraph()

#train this evolved pynet on the dataset
training_result = evolved_pynet.Train(X,T)

#simulate the output of the network given input data
Y = evolved_pynet.Sim(X)
```

# DNA
How is a Pynet able to evolve exactly? It begins with the idea that representing an object with "DNA" is simply an abstraction of how a computer holds data. It means that zeros and ones store the information in a structured pattern. This pattern may be applied in order to represent every single permutation the object may exist as.
```python
#extract dna from the network
dna = Pynet.DNA.Extract(evolved_pynet)

#re-form the network from the extracted dna
formed_net = Pynet.DNA.Form(dna,groupSize+10,useGPU) #prove groupsize isnt part of DNA

```

# Data Preparation

The quality of Pynet you create is very dependant on the datasets used and the type of problem addressed. 
To prepare these datsets for using with a Pynet you must follow the criteria:
* Must be a <strong>np.array</strong>
* X input dataset has the shape -> [n_samples, n_features]
* T target dataset has the shape -> [n_samples, n_classes]


# Saving/Loading a Pynet
There are fundamentally 2 ways to store a Pynet. Each method has its advantages and disadvantages.
### Using DNA
Saving/loading a pynet using DNA <strong>does not store weights/bias</strong> and other information unique to the members of the group. This means a network group stored/retrieved in this manner will need to be retrained before it may perform.
```python

#save the raw dna as text
with open('evolved_dna.txt','w') as text_file:
    text_file.write(str(dna))

#load a network from dna
dna = None
with open('evolved_dna.txt','r') as text_file:
    dna = int(text_file.read())

#reform the network
formed_net = Pynet.DNA.Form(dna,groupSize,useGPU)

```

### Dictifying
This process transforms the network into a dictionary then into json text. This makes reusing a specific network possible.
```python
import json

#save net as a dict in json format
with open('evolved_net.json','w') as outfile:
    outfile.write(json.dumps(evolved_net.Dictify()))

#load
evolved_net = None_
with open('evolved_net.json','r') as outfile:
    evolved_net = Pynet.IO.Load(outfile.read(),useGPU)

```

# Tips

Changing a pynet's structure should be done before configuration
```python
#change a fitnet
fitnet = Pynet.Models.FitnetGroup(groupSize,useGPU)

#into a cascade
fitnet.isInput[-1] = True

#with a different final transfer
fitnet.layers[-1].transferFcn = 'swish'

#then configure
fitnet.ConfigureGraph()

```
Trained a very large group but only need a single network from it?
```python
#pick a group member to get
member_idx = 7

#get the member out
member = Pynet.MemberOut(nnet,member_idx)

#they can still simulate a signal output
Y = member.Sim(X)
```

# Further Information
Looking for more explanation? The code itself is documented in a way meant to be read and understood easily. A good place to start with this package would be:
```python
pynet.Tests.main
```

# TODO
 Currently there are yet to be implimented features:
 * Complete recurrent layer capability so sequential networks may be evolved, such has LSTM
 * Higher dimensionality of layers must be added in order to support typical computer vision applications such as 2-D Convolutions


 ##### fin

