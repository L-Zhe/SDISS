# Neural Sentence Simplification with Semantic Dependency Information

Code for paper [Neural Sentence Simplification with Semantic Dependency Information by Zhe Lin, Xiaojun Wan](https://ojs.aaai.org/index.php/AAAI/article/view/17578). This paper is accepted by AAAI'21. Please contact me at [linzhe@pku.edu.cn](mailto:linzhe@pku.edu.cn) for any question.

## Structure

<img src="https://github.com/L-Zhe/SDISS/blob/main/img/overview.jpg?raw=true" width = "600" alt="overview" align=center />

## System Output

If you are looking for system output and don't bother to install dependencies and train a model (or run a pre-trained model), the `Result` folder is for you.

## Dependencies

```undefined
PyTorch 1.4
NLTK 3.5
stanfordcorenlp
tqdm 4.59.0
```

We provide *SARI* metric in our code, you need to set up according to [here](https://github.com/XingxingZhang/dress/tree/master/experiments/evaluation/SARI), and change the path in ``eval.py``. 

## Datasets

We provided the original and  preprocessed datasets on [release page](https://github.com/L-Zhe/SDISS/releases/tag/1.0), include *Newsela*, *WikiSmall* and *WikiLarge*. You can also get them on [*Newsela*](https://newsela.com), [*WikiSmall* and *WikiLarge*](https://github.com/louismartin/dress-data/raw/master/data-simplification.tar.bz2)

8 references *WikiLarge* test set can be downloaded [here](https://github.com/cocoxu/simplification/tree/master/data/turkcorpus).

Note that, the copyright of the *Newsela* belongs to  https://newsela.com, we only provide it as research apporaches. **For any purpose of using these datasets beyond academic research please contacts newsela.com.**

## Preprocess

We provide all preprocessed data of *Newsela*, *WikiSmall* and *WikiLarge* on [release page](https://github.com/L-Zhe/SDISS/releases/tag/1.0). 

In order to reduce the vocabulary size, we tag words with their named entities using the [Stanford CoreNLP tool](https://stanfordnlp.github.io/CoreNLP/) (Manning et al. 2014), and anonymize with NE@N token, N indicates NE@N is the N-th distinct NE typed entity. If you want to leverage your own datasets, please employ ``NER.py`` to replace *Named-entity* in the sentence. You must provide SDP graph of datasets yourself in accordance with the format of SDP graph data in the file.

## Train

All configurations of the training step are shown in the ``Parameter.py``. We will generate the validation data and evaluate the result after each epoch training.  

Changing the **mode** in ``Parameter.py`` into **train**, then run the following code to start training model.


```python
python main.py
```

## Inference

Change the **mode** in ``Parameter.py`` into **test** to begin inference.

We provide pre-trained models of three benchmark datasets [release page](https://github.com/L-Zhe/SDISS/releases/tag/1.0).

## Evaluation

We provide *SARI*, *BLEU* and *FKGL* evaluation metrics in our code. 

Notice that, due to the problem of encoding of python2 and python3, the FKGL we provided may be a bit different from the previous version and can only as a reference. We final calculate the FKGL score followed [here](https://github.com/XingxingZhang/dress/tree/master/dress/scripts/readability) on python2.

Our code only provide SARI with one reference. The WikiLarge which containing with 8 references should be evaluated as [here](https://github.com/XingxingZhang/dress/tree/master/experiments/evaluation/SARI).

## Results

<img src="https://github.com/L-Zhe/SDISS/blob/main/img/result.jpg?raw=true" width = "400" alt="overview" align=center />



## Reference

If you use any content of this repo for your work, please cite the following bib entry:

```
@article{lin2021simplification,
  title={Neural Sentence Simplification with Semantic Dependency Information},
  author={Zhe Lin, Xiaojun Wan},
  journal={AAAI Workshop on Deep Learning on Graphs: Methods and Applications},
  year={2021}
}
```
