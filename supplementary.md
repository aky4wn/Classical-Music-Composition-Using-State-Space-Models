---
layout: page
title: Supplementary Materials
permalink: /supplementary/
---
Supplementary materials for the paper "Classical Music Composition Using State Space Models".

## General Trends of Composed Pieces
[Plot]({{ site.url }}/supp/muss_RMSE_supp.pdf) of the RMSE for various metrics for all 14 original models when trained on Pictures at an Exhibition, Promenade - Gnomus.

[Plot]({{ site.url }}/supp/muss_info.pdf) of the average mutual information and average minimum edit distance for all 14 original models when trained on Pictures at an Exhibition, Promenade - Gnomus.

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
