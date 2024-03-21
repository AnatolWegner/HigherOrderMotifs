Dependencies: graph-tool (https://graph-tool.skewed.de), igraph (https://python.igraph.org/en/stable/)

The folder contains pickled files that contain sets of motifs. e.g. 'Umotifs6' : undirected motifs up to 6 nodes, 'Dmotifs4' : directed motifs up to 4 nodes etc. 

InferC.py contains all functions for inferring subgraph configurations based on various models. 

The notebook run.ipynb contains an example for inferring subgraph configurations. 

The notebook AnalyseC contains functions for plotting subgraph configurations and motifs, and functions for description lengths of various models. 

The notebook SignificanceProfiles contains functions for calculating significance profiles for configurations. 

The notebook RandomizeDCSGCM contains functions for randomizing configurations and generating configurations for various models. 

The notebook GenerateM contains functions for generating various sets of motifs (based on nauty). 

The Results folder contains pre-computed MAP configurations for various network analysed in the paper based on degree corrected and non-DC models. 

