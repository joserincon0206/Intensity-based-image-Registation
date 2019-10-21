# Intensity-based Image registration

Details: 

1. The goal is to align two images assuming that the missaligment can be recovered with an affine transformation.  
we also assumed that the pixel intensity between an object in the two images is at most contaminated by a small Gaussian noise. 
I.e., we can use L2 norm as our cost function. 

2. Write from scratch the non-linear optimization algorithm using gradient descent. 
