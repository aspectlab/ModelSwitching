#ModelSwitching

This code repository contains code used to generate results in this paper --

A. Grootveld, K.P. Vedula, V. Bugayev, L. Lackey, D.R. Brown, and A.G. Klein, “Tracking of dynamical processes with model switching using temporal convolutional networks,” in Proc. IEEE Aerospace Conference, Mar. 2021.

For each of the two examples (GilbertElliot and ManeuveringTargets), the process for repeating results in the paper is as follows:

1. In Matlab: Run main.m, which generates several gigabytes of data in the data folder.
2. In Python: Run main_tcn.py to generate data/test_tcn.mat.  Note that the default settings use a pre-trained model (i.e., the one used in the paper), but by setting a flag you may instead train your own model which should achieve similar performance.
3. In Matlab: Run plotresults.m to plot results.

Note that Matlab or Octave is required in steps 1 and 3, while step 2 requires Python with Keras as well as this TCN package -- https://github.com/philipperemy/keras-tcn

Authors: A. Grootveld, L. Lackey, A.G. Klein

