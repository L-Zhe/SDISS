# SDISS

Code for paper Neural Sentence Simplification with Semantic Dependency Information by Zhe Lin, Xiaojun Wan. This paper is accepted by AAAI'21. Please contact me at [linzhe@pku.edu.cn](tomail:linzhe@pku.edu.cn) for any question.

## Dependencies

```undefined
PyTorch 1.4
NLTK 3.5
stanfordcorenlp
tqdm 4.59.0
```

## Datasets

We provided the original and  preprocessed datasets on [release page](https://github.com/L-Zhe/SDISS/releases/tag/1.0), include *Newsela*, *WikiSmall* and *WikiLarge*. You can also get them on [*Newsela*](https://newsela.com), [*WikiSmall* and *WikiLarge*](https://github.com/louismartin/dress-data/raw/master/data-simplification.tar.bz2)

8 references *WikiLarge* test set can be downloaded here https://github.com/cocoxu/simplification/tree/master/data/turkcorpus

Note that, the copyright of the *Newsela* belongs to  https://newsela.com, we only provide it as research apporaches. **For any purpose of using these datasets beyond academic research please contacts newsela.com.**

## Preprocess

We provide all preprocessed data of *Newsela*, *WikiSmall* and *WikiLarge* on [release page](https://github.com/L-Zhe/SDISS/releases/tag/1.0). 

In order to reduce the vocabulary size, we tag words with their named entities using the [Stanford CoreNLP tool](https://stanfordnlp.github.io/CoreNLP/) (Manning et al. 2014), and anonymize with NE@N token, N indicates NE@N is the N-th distinct NE typed entity. If you want to leverage your own datasets, please employ *NER.py* to replace *Named-entity* in the sentence. You must provide SDP graph of datasets yourself in accordance with the format of SDP graph data in the file.

## Training

All configuration of training step are shown in the *Parameter.py*. We will generate the validation data and evaluate the result pre epoch. In our code, we have provide *BLEU* and *FKGL* metrices. For *SARI*, you need to set up according to [here](https://github.com/XingxingZhang/dress/tree/master/experiments/evaluation/SARI), and change the path in *eval.py*

Notice that, due to the problem of encoding, the FKGL we provided may be a bit different with the previous version and can only as a reference. We final calculate the FKGL score followed [here](https://github.com/yuedongP/EditNTS/blob/master/utils/fkgl.py) on python2.

## Test

We provide all models of three benchmark and the correspond results on [release page](https://github.com/L-Zhe/SDISS/releases/tag/1.0).