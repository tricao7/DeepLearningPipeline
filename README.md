# DeepLearning_with_MongoDB - Multi-Label Animal Classification

## Table of Contents

1. [Introduction](#introduction)
2. [Data](#data)
3. [MongoDB](#mongodb)
4. [Model](#model)

## Introduction

A demonstration of how to supplement Modern Deep Learning with MongoDB's Cloud Services.

*{Insert more details here}*

## Data

The [Multi-Label Animal Classification](https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset) data we will be using can be found on kaggle. The Training Data consists of 15 clases (Beetle, Butterfly, Cat, Cow, etc.) with 2000 images each, while the Validation Data has 100-200 per class. The images are in standard .jpg format and stored in class folders.

## MongoDB

Utilizing MongoDB's Cloud Services, we will be able to store and access the data in a more efficient manner. To prepare the data for MongoDB, we will vectorize the images and store them as csv files.

## Model

For our multi-label classification model, we will be using PyTorch along with; the Hugging Face's [Transformers](https://huggingface.co/transformers/) library. The model we will be using is the [ResNet50](https://huggingface.co/microsoft/resnet-50) model, which is a pre-trained model that has been trained on ImageNet.

```python
import torch.nn as nn
from transformers import ResNetForImageClassification
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
```

We have alternated the last layer of the model to fit our 15 classes instead of the original 1000 classes.

```python
model.classifier[1] = nn.Linear(in_features=2048, out_features=15, bias=True)
```

## Conclusion

*{Insert more details here}*


## References

1. [Multi-Label Animal Classification](https://www.kaggle.com/datasets/utkarshsaxenadn/animal-image-classification-dataset)
2. [Transformers](https://huggingface.co/transformers/)
3. [ResNet50](https://huggingface.co/microsoft/resnet-50)
