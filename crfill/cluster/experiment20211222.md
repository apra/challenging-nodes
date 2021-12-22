The VAE model is still quite blurry.
To improve it, I can add some of the tricks used by the NVAE paper.

Preliminary results show that low downsample is beneficiary. Make sure
that the code is actually doing less downsamplings, and then try with 0 downsamplings
and 1 downsampling.

Parameters to look at: downsample, 0,1,2.

Then, the effect of sigma can be studied, specifically decreases the variance.

The network can also be made bigger. **Add a parameter to add layers**.

Change number of filters used initially, always scale them in the same way by doubling.

