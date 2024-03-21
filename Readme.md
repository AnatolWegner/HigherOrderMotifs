Dependencies: graph-tool (https://graph-tool.skewed.de), igraph (https://python.igraph.org/en/stable/)

The folder contains pickled files that contain sets of motifs. e.g. 'Umotifs6' : undirected motifs up to 6 nodes, 'Dmotifs4' : directed motifs up to 4 nodes etc. 

The jupyter notebook AnalyseC contains functions for plotting subgraph configurations and motifs, and functions for description lengths of various models. 

The notebook SignificanceProfiles contains functions for calculating significance profiles for configruations. 

The notebook RandomizeDCSGCM contains functions for randomizing configurations and generating configurations for various models. 

The notebook GenerateM contains functions for generating various sets of motifs (based on nauty). 

The Results folder contains pre-computed MAP configurations for various degree corrected and non-DC models. 

