# Split-GPLVM

## Intro 
This is a model that builds a mixture of Gaussian Process Latent Variable Model (GPLVM), with a kernel that splits the latent space into a shared space and a private space that is specific to each mixture. Inference is done by extending the variational inducing point framework previously proposed for GPLVM. A very detailed writeup of the model, including derivation of the variational objective, is in `Split-GPLVM.pdf`.

Currently, the model appears to do the following correctly ([slides](https://docs.google.com/presentation/d/1yYXcHC3vwZJRUM-RJDr6heKQP_GP1tQHtJcqTfxbP7w/edit?usp=sharing)):
1. It reproduces the Overlapping Mixture of Gaussian Process (OMGP) when it's given the true latent embedding (`X`) and only required to learn the mixture. 
2. It recovers the correct ELBO for GPLVM when fitted with one mixture component and taking one natural gradient step.

But the model does not work when learning both the mixture assignment and latent embedding simultaneously ([slides](https://docs.google.com/presentation/d/1jDvBdd4FvmJpQ4PTOkiCiRiR5uqVuRSQv5p1APhO8j0/edit?usp=sharing)) We think it's due to an unidentifiability issue. Essentially, the model is too flexible, and we have not come up with a sensible way to constrain the model. A more detailed explaination of the issue is below:

On a high level, if the data can be separated into distinct groups, GPLVM can use one dimension to separate those points, and explain the observed well (i.e. explain through embedding). But for our model, we want to the model to use the mixture to capture difference between distinct groups (i.e. explain through mixtures), then use each GPLVM to explain the variance within that group. However, from the model’s point of view, there is no way to determine which mechanisms of explanations it should use. So different initializations (e.g. initializations of pi, the mixture weights) will lead the model to do different things, and either way will explain the data well. 

## Files
`split_gplvm.py` is the main implementation of the model. It is largely based on the GPLVM implementation in `GPflow`. 

`utils.py` and `plot.py` contain various miscellaneous functions used to debug the model (without much success..)