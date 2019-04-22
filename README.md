# demo-necker-cube

this is a recurrent neural network for the 
<a href="https://en.wikipedia.org/wiki/Necker_cube">Necker cube</a> 
bistable perception phenomenon

play with the model on google colab: 
<a href="https://colab.research.google.com/github/qihongl/demo-necker-cube/blob/master/necker_rnn.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open in Colab" title="Open and Execute in Google Colaboratory">
</a>


here's the model [1]: 

<img src="https://web.stanford.edu/group/pdplab/pdphandbook/CubeDiagram.png" alt="necker">



the figure below shows the network internal states over time, each line represents the activity time course of one neuron, and the two colors represent the two interpretations

<img src="https://github.com/qihongl/demo-necker-cube/blob/master/imgs/temp_dyn.png" alt="temporal dynamics of necker" width=600>


here's the connection weights of this network: 
<img src="https://github.com/qihongl/demo-necker-cube/blob/master/imgs/wts.png" alt="w of necker" width=700>


### reference: 
[1] Constraint Satisfaction in PDP Systems. (2015, December 16). Retrieved April 21, 2019, from https://web.stanford.edu/group/pdplab/pdphandbook/handbookch4.html
