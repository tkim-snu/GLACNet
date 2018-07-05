# GLACNet

This repository is the implementation of [GLocal Attention Cascading Network](https://arxiv.org/abs/1805.10973) for the [Visual Storytelling Challenge](http://www.visionandlanguage.net/workshop2018) as a team SnuBiVtt.

![Architecture of GLocal Attention Cascading Network](misc/architecture.jpg)

### Prerequisites
#### 1. Download 'punkt'
```{.python}
nltk.download('punkt')
```

### Preprocessing


### Training & Validation

```
python train.py
```

### Evaluation

```
python eval.py --num_model my_model_num
```
The result.json file will be found in the root directory.

### Citation

```
@article{Kim2018,
author = {Kim, Taehyeong and Heo, Min-Oh and Son, Seonil and Park, Kyoung-Wha and Zhang, Byoung-Tak},
journal = {arXiv preprint arXiv:1805.10973},
title = {{GLAC Net: GLocal Attention Cascading Networks for Multi-image Cued Story Generation}},
year = {2018}
}
```

### License

MIT License
