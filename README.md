# Align-mmt

<!-- []() -->

## Requirements:
- pip install Stanza 1.2.2 Stanza_batch 0.2.2
-`pip install --editable .` or `python setup.py build_ext --inplace`
- conda install pytorch==1.7.0 torchvision==0.8.0 cudatoolkit=10.1 -c pytorch
- Python 3.7.6
- Java JDK 1.8.0 (or higher)
- Meteor-1.5.tar.gz

##　Data Preparation


## Resources
File Name | Description |  Download
---|---|---
`resnet50-avgpool.npy` | pre-extracted image features, each image is represented as a 2048-dimensional vector. | [Link](https://1drv.ms/u/s!AuOGIeqv1TybbQeJMw8CdqOphfA?e=l8k4df)
`en-de` | BPE+TOK text, Image Index, Label for English-German task (including train, val, test2016/17/mscoco)
`en-fr` | BPE+TOK text, Image Index, Label for English-French task (including train, val, test2016/17/mscoco)