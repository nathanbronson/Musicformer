<p align="center"><img src="https://github.com/nathanbronson/Musicformer/blob/main/logo.jpg?raw=true" alt="logo" width="200"/></p>

_____
# Musicformer
***ENTIRELY SOMETHING***

## About
*Musicformer is a work-in-progress. The existing models are prone to collapse. Additional work is needed to stabilize the models. The vector space model is also not yet integrated with the music interface.*

Musicformer is a neural network for embedding music data in a linear manifold for comparison and analysis. This repository contains two experiments from this project's development.

The first experiment developed is housed in `VAETransformer/`. This experiment involved augmenting a convolutional VAE with a transformer. The model applied variational reparameterization between the transformer's encoder and decoder. To have the desired dimension-reduction effect, the encoder's output was downsampled significantly.

This experiment showed some promising results. The embeddings produced by the model showed some recognition of high-level qualities of a song's sound and feel, but the results were not as apparently granular as hoped. The model was prone to collapse because its decoder was autoregressive. Several model variations were introduced to address this, including encoder and decoder only models with bottlenecks and augmented vanilla AE. These models showed mixed results.

The second experiment developed is housed in `VectorSpaceAutoEncoder/` (`vsae`). This experiment designed a model following the vector space axioms. Specifically:
> Take the vector space ${\mathbb{R}^n}$ over vector addition and scalar multiplication. \
Now, take a set ${S}$ with a vectorization map ${v : S \xrightarrow{bij} V \subseteq \mathbb{R}^n}$. \
For an arbitrary ${v}$, 
it is possible (but rather unlikely depending on the nature of ${S}$ and ${v}$) that ${\exists}$ ${ f : V \xrightarrow{bij} \mathbb{R}^{m \leq n}}$. \
It is more likely, though, that there exists an ${f'}$ that resembles an ${f}$ with a bounded, contiguous codomain ${D \subseteq \mathbb{R}^{m \leq n}}$. \
Regarding this resemblance, ${\exists}$ ${C \subseteq D, A \subseteq \mathbb{R}}$ where ${\forall c_1, c_2 \in C, c_1 + c_2 \in D, a c_1 \in D, \exists}$ ${w \in V : f(w) = c_1}$. \
${f'}$, then, can be described as ${f' : V \xrightarrow{inj} D}$ (having a corollary map ${V \supseteq V' \xrightarrow{bij} C \subseteq D}$), and the optimal ${f'}$ maximizes ${|C|}$. \
For a more optimal ${f'}$, then, ${D}$ will like a vector space over vector addition and scalar multiplication for more elements. \
We can now define ${\mathcal{E} : S \rightarrow D}$ as ${f \circ v}$, creating an encoder function (with its approximation ${\mathcal{E}'}$ being ${f' \circ v}$) that embeds an arbitrary set ${S}$ into a vector space. \
To create a decoder, we invert the encoder to get ${\mathcal{D} : D \rightarrow S}$ as ${v^{-1} \circ f^{-1}}$. \

> To create approximate ${\mathcal{E}}$ and ${\mathcal{D}}$ by gradient descent, we must find a differentiable proxy for ${|C|}$. \
Because ${c \in C \longleftrightarrow (\mathcal{E}' \circ \mathcal{D}')(c) = c}$ ${\land}$ ${\exists}$ ${s \in S : \mathcal{E}'(s) \in C}$, our loss should be a combination of a reconstruction loss and a discrimination loss. \
In a vector space, distance or a monotonic transformation of it will be a sufficient reconstruction loss, and we can formulate a discrimination loss ${\mathcal{C} = p(\mathcal{D}(c) \in S)}$, which can be approximated by minimizing ${BCE(\mathcal{C}'(0, S))}$ and ${BCE(1, \mathcal{C}'(1, \mathcal{D}'(c)))}$\
We can, therefore, formulate the differentiable proxy ${\mathcal{L} = MSE(c, (\mathcal{E}' \circ \mathcal{D}')(c)) + \mathcal{C}(\mathcal{D}(c))}$

To simplify calculations, we partition the vsae loss, such that the decoder is trained by the discrimination loss and the encoder by the reconstruction loss. The effect is a model resembling a low-dimensional GAN with an auxilliary network trained to predict the input noise from the output.

This model was tested on MNIST data. It also exhibited collapse, causing the discriminator to outpace the decoder even when weakened and selectively trained.

## Usage
The scripts to run the various experiments are available in each experiment's directory. These two models have not yet been integrated, so there is not yet a main script.

## License
See `LICENSE`.