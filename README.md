##iPhone Image Classifier (Transfer Learning)

This project is me taking a real shot at learning PyTorch and applying transfer learning in a practical way.
As a starting point, I built a simple image classification model that uses a pretrained EfficientNet-B0 to recognize whether an image contains an iPhone or not.

The goal here wasn’t to build something production-ready, but to understand the full pipeline: loading a pretrained model, freezing weights, replacing the classifier, training on custom data, saving/loading model state, and running inference.

This is part one of a larger idea, more work and experimentation will be added as I continue learning and building on top of this foundation.

Tech used:

PyTorch

torchvision / EfficientNet-B0

Transfer learning

Apple Silicon (MPS)
