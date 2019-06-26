![](img/logo3.png)

# Exploring latent attitudes using word2vec
> Uncovering implicit bias using domain-specific word embeddings

Vectorized word embeddings trained on a large corpus such as Wikipedia or Google News can be used to reveal relationships between pairs of words. For example, the vector difference between the words "Paris" and "France" is very similar to the vector difference between the words "Tokyo" and "Japan". This feature of word embeddings can be used to reveal not only factual relationships, but also conceptual relationships relating to gender, race or any other dimension. However, they often reflect a degree of implicit bias that can result in unfortunate stereotypes. For instance, Bolukbasi discovered that by applying the difference between man and woman to the phrase 'computer programmer' the resulting vector was most similar to ['homemaker'](https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf)

By using broad-based corpora as the source for embeddings, researchers in this area are uncovering widespread biases that are present in society as a whole. This project seeks to examine bias more narrowly, by comparing word embeddings that are derived from individual domains.

Specifically, we'll look at embeddings based upon;

 * the works of male authors
 * the works of female authors
 * scripts of Hollywood movies
 * lyrics of the top 100 songs from the US and the UK over each of the last fifty years

 to see whether certain stereotypes are universal, or whether they differ depending upon the set of authors.

## Structure of the repo

The project is split into three sections

**A. Data Collection**

[Data](1_Data_Collection.ipynb) is sourced from sites that are publicly accessible and free of charge. The downside is that multiple sources are required to obtain a substantially complete set of song lyrics, meaning that there is no consistent format for either the HTML or the author / title combinations. A lot of data cleansing is required, and a different script is needed to scrape each site.

**B. Initial exploration**

This includes an [examination](2_Initial_Exploration.ipynb) of corpus size and word frequencies, including demonstration of Heap's Law and Zipf's Law. Initial embeddings are calculated using a basic skip-gram model in TensorFlow, and then projected into 2-D space using t-SNE for the purposes of visualizing clusters. For the more detailed analysis in part (c), embeddings are calculated using the gensim implementation of word2vec.

**C. Comparative analysis**

Embeddings are compared in two ways. [First](3_Examination_of_cosine_similarity_pairs.ipynb), pairs of words that display a high degree of cosine similarity in one dataset but not in another are identified. This gives a sense of some of the topics or concepts that are uniquely prominent in a given dataset.

The [next](4_Examination_of_Word_Associations.ipynb) section is based upon work done in social psychology to uncover implicit bias. Defined sets of conceptual words (for example, words associated with freedom, science, weapons etc) are examined alongside specific attributes (for example, gender) or subjective measures (such as pleasantness). This reveals specific instances of latent bias, some of which may have been held by the author, or some of which may have been a reflection of societal values.

A selection of papers that inspired the project can be found [here](Papers) 

### Summary findings

* Despite the idiosyncratic linguistic nature of song lyrics (where sentences are brief and repetitive, and words may be chosen mainly because they rhyme) embeddings still capture recognizable relationships. For example, the five words most similar to "man" are "woman", "guy", "kid", "fool" and "wife". Those most similar to "music" are "rhythm", "radio", "sound", "dj" and "funky".
* A simple frequency count reveals differences in topics. After eliminating stop words, "love" is the most frequent word used in song lyrics. It appears in 66th place in the works of female authors, but not until 205th place in those of male authors. The word "marriage" ranks roughly 1200 places higher in the female data than in the male data. The converse is true for the word "war".
* When looking at cosine similarity pairings that are significant in one set only, there are eight times as many such pairs in the works of female authors versus movie scripts than in male authors versus movie scripts. This may be because females authors use more varied combinations of words than male authors, or it could be because male authors use pairings that are more consistent with those found in movie scripts.
* Commonly held stereotypes persist in the specific datasets, but the degree of association can vary. While male nouns and pronouns are associated with math / science, and female terms with the arts, the relationship is weak in the works of male authors. It is actually strongest in the works of female authors.
* Female authors also tend to associate the concept of freedom with male terms, more so than male authors do. The strongest correlation between freedom and male terms appears in movie scripts.
* All datasets apart from song lyrics display signs of gender bias when discussing illness. Physical illness is slightly more associated with female terms, but mental conditions (ranging from sadness to depression) are much more likely to be associated with female terms.
* Female authors typically view mental afflictions as temporary, suggesting that recovery is possible. Male authors make no real distinction between the temporary or permanent nature of disease. Movie scripts however, tend to strongly associate physical illness with permanence, suggesting that characters with a physical illness will not recover.

### Required packages

As well as common packages, such as NumPy, pandas, re etc, the workbooks use

* TensorFlow
* nltk
* gensim
* sklearn
* scipy
* Bokeh
* Beautiful Soup


_Optional_

* bhtsne

t-SNE calculations are performed using scikit-learn, although this can require significant amounts of memory. If this is an issue, instructions for installing the [bhtsne](https://github.com/lvdmaaten/bhtsne) package are included. This uses Barnes-Hut t-SNE, which is an efficient approximation of t-SNE, running in _O(NlogN)_ time and requiring _O(N)_ memory, instead of _O(N^2)_ for the full implementation.

