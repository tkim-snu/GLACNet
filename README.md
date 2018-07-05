# GLACNet

This repository is the implementation of [GLocal Attention Cascading Network](https://arxiv.org/abs/1805.10973) for the [Visual Storytelling Challenge](http://www.visionandlanguage.net/workshop2018) as a team SnuBiVtt.

![Architecture of GLocal Attention Cascading Network](misc/architecture.jpg)

<br>

### Prerequisites

##### 1. Clone the repository
```
git clone https://github.com/AcousticRicky/GLACNet.git
cd GLACNet
mkdir models
```

##### 2. Download requirements
```
pip install -r requirements.txt
```

##### 3. Download 'punkt'
```{.python}
nltk.download('punkt')
```

<br>

### Preprocessing

##### 1. Download the dataset
[VIST hompage](http://visionandlanguage.net/VIST/dataset.html)

##### 2. Resize Images
```
python resize.py
```

<br>

### Training & Validation

```
python train.py
```

<br>

### Evaluation

```
python eval.py --model_num my_model_num
```
The result.json file will be found in the root directory.

<br>

### Citation

```
@article{Kim2018,
author = {Kim, Taehyeong and Heo, Min-Oh and Son, Seonil and Park, Kyoung-Wha and Zhang, Byoung-Tak},
journal = {arXiv preprint arXiv:1805.10973},
title = {{GLAC Net: GLocal Attention Cascading Networks for Multi-image Cued Story Generation}},
year = {2018}
}
```

<br>

### License

MIT License
