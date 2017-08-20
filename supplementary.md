---
layout: page
title: Supplementary Materials
permalink: /supplementary/
---
Supplementary materials for the paper "Classical Music Composition Using State Space Models".

We considered 15 models, M1-M15, to generate compositions: model M1
was a standard HMM with 25 hidden states, model M2 was a 2-HMM with
25 hidden states, M3 was a 3-HMM with 10 hidden states, model M4 was a
LRHMM with 25 hidden states, model M5 was a 2-LRHMM with 25 hidden
states, model M6 was a 3-LRHMM with 10 hidden states, model M7 was
a ARHMM with 25 hidden states, model M8 was a HSMM with 25 hidden
states, model M9 was a NSHMM with 25 hidden states, M10 was a TSHMM
with 10 hidden states in the rst layer and 5 in the second layer, M11 was a
TSHMM with 5 hidden states in the rst layer and 10 in the second layer,
model M12 was a FHMM with three independent HMMS each with 15,
10 and 5 hidden states respectively, model M13 was a LHMM with three
layers and 25 hidden states in each layer, model M14 was a TVAR with
order between 7 and 14 and model M15 was a baseline model specied by a
HMM with randomly assigned parameters.


## General Trends of Composed Pieces
[Plot]({{ site.url }}/supp/muss_RMSE_supp.png) of the RMSE for various metrics for all 14 original models when trained on Pictures at an Exhibition, Promenade - Gnomus.

[Plot]({{ site.url }}/supp/muss_info.png) of the average mutual information and average minimum edit distance for all 14 original models when trained on Pictures at an Exhibition, Promenade - Gnomus.

[Table]({{ site.url }}/supp/muss_RMSE_table.pdf) of the RMSE for various metrics for all 14 original models ordered from best to worst when trained on Pictures at an Exhibition, Promenade - Gnomus.


## Validation of Trends
In addition to ranking each piece from favorite to least favorite for the second round of evaluations, each listener was asked to quantitatively score each piece on a scale of 1 to 5 according to:
- how much the generated piece sounded like it was composed by a human, 1 = piece sounded completely random, 5 = piece sounded just like a human composition;
- how harmonically pleasing each piece was, 1 = not at all harmonically pleasing, 5 = very harmonically pleasing;
- how melodically pleasing each piece was, 1 = not at all melodically pleasing, 5 = very melodically pleasing.

The average rankings for each evaluated piece are [here]({{ site.url }}/supp/rankings_table.pdf)

The [percentage]({{ site.url }}/supp/simple_harmonic.pdf) of simple harmonic intervals that are thirds, perfect fourths or fifths and dissonant for the second round of evaluated pieces.

## TSHMM Baum-Welch Algorithm
The [Baum-Welch Algorithm]({{ site.url }}/supp/appendix.pdf) for the two hidden state HMM.
