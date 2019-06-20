from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import math
import nltk
import collections
import random
import os
import sys
import argparse
import re
from six.moves import xrange
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.contrib.tensorboard.plugins import projector
from nltk.corpus import stopwords
nltk.download('stopwords')


def tokenize(clean, exclude_stopwords=True):
    """
    Applies NLTK tokenizer to a document, removes stop words by default
    and returns a list of all remaining words along with the size of 
    the vocabulary (ie distinct words)
    """
    tokens = nltk.word_tokenize(clean)
    text = nltk.Text(tokens)   
    if exclude_stopwords:
        stop_words = set(stopwords.words('english'))
        words = [w for w in text if not w in stop_words]
    else:
        words = [w for w in text]
    vocab = sorted(set(words))
    print("Document contains ", len(vocab),"distinct words")
    print('Total word count = ', len(words))
    return words, len(vocab)


def build_dataset(words, n_words):
    """Process list of words into useful data structures of a given size
       Selects only the n_words most frequent words and discards the rest
    
    Args:
        words: list of words to be processed
        n_words: desired size of ouput vocabulary
        
    Returns:
        data: List where each word in the text has been replaced with an integer representation
              of how it ranks in terms of frequency in the corpus, so the most frequent word is
              replaced by 1, the second most frequent by 2 etc. Any words less frequent that the
              limit set by n_words are represented by zero
        count: Ordered list of the most frequent words, where each item is a tuple of the word
               itself, and the frequency with which the word occurs in the corpus
        dictionary: Dictionary where k,v pairs are word : frequency rank
        reversed_dictionary: Dictionary where k,v pairs are frequency rank : word 
        """
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0:  # ie dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def generate_batch(data, batch_size, num_skips, skip_window):
    """
    Creates batches for use in training a basic skip-gram model. Depends on the variable
    "data" having been already set using the previously defined function build_dataset()
    and also assumes that data_index has already been initialized
    
    Args:
        batch_size: number of words in the mini-batch, taken iteratively from the list "data"
        which is returned by the function build_dataset
        num_skips: how many times to re-use an input to generate a label
        skip_window: size of the window from which context words may be taken
    
    Returns:
        batch: slice of the list 'data', ie a list based on the original text where each word has been
        replaced by an integer representing the frequency rank of that word
        labels: randomly selected word from the context window, again replaced by the integer that represents
        the frequency rank of that word.
        Note that the frequency rankings can be used as keys in "reversed_dictionary" to recover the original word
    """
    #global data
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    
    # create empty arrays of the right size, and an empty buffer using deque (double ended queue)
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1
    buffer = collections.deque(maxlen=span)
    
    if data_index + span > len(data): #here 'data' is the output from build_dataset()
        data_index = 0
    # add a slice of the data to the buffer and move the index along
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        # cycle through the words randomly selected from the context window and
        # add the original word to the batch and the context word to the labels
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0:span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
        # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels


def build_graph_and_train(data, batch_size=100, embedding_size=300, skip_window=1, num_skips=2, num_sampled=20, vocabulary_size=10000):
    """
    Set up TensorFlow graph and paths for Tensorboard, and then create embeddings using the defined hyperparameters
    
    Returns an array of size vocabulary * dimensions, where vocabulary = n_words passed to function build_dataset()
    and dimensions = embedding_size.
    
    Each row is the embedding vector for the corresponding word in the vocabulary
    """
    #global data
    global data_index
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    graph = tf.Graph()

    with graph.as_default():

      # Input data.
        with tf.name_scope('inputs'):#2
            train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
            train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

      # Assumes that we're using a CPU
        with tf.device('/cpu:0'):
        # Look up embeddings for inputs.
            with tf.name_scope('embeddings'):
                embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
                embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
            with tf.name_scope('weights'):
                nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size],
                                                              stddev=1.0 / math.sqrt(embedding_size)))
            with tf.name_scope('biases'):
                nce_biases = tf.Variable(tf.zeros([vocabulary_size]))#6

      # Compute the average NCE loss for the batch.
      # tf.nce_loss automatically draws a new sample of the negative labels each
      # time we evaluate the loss.
      # Explanation of the meaning of NCE loss:
      #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
        with tf.name_scope('loss'):
            loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                                 biases=nce_biases,
                                                 labels=train_labels,
                                                 inputs=embed,
                                                 num_sampled=num_sampled,
                                                 num_classes=vocabulary_size))

      # Add the loss value as a scalar to summary.
        tf.summary.scalar('loss', loss)

      # Construct the SGD optimizer using a learning rate of 1.0.
        with tf.name_scope('optimizer'):
            optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)#4

      # Compute the cosine similarity between minibatch examples and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
        similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

      # Merge all summaries.
        merged = tf.summary.merge_all()

      # Add variable initializer.
        init = tf.global_variables_initializer()

      # Create a saver.
        saver = tf.train.Saver()
    current_path = os.path.dirname(os.path.realpath(sys.argv[0]))

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir',type=str,default=os.path.join(current_path, 'log'),
                        help='The log directory for TensorBoard summaries.')
    FLAGS, unparsed = parser.parse_known_args()

    # Create the directory for TensorBoard variables if there is not.
    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    num_steps = 100001

    with tf.Session(graph=graph) as session:
      # Open a writer to write summaries.
        writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)#2

      # We must initialize all variables before we use them.
        init.run()
        print('Initialized')

        average_loss = 0
        data_index = 0
        for step in xrange(num_steps):
            batch_inputs, batch_labels = generate_batch(data, batch_size, num_skips,skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # Define metadata variable.
            run_metadata = tf.RunMetadata()

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        # Also, evaluate the merged op to get all summaries from the returned "summary" variable.
        # Feed metadata variable to session for visualizing the graph in TensorBoard.
            _, summary, loss_val = session.run([optimizer, merged, loss],
                                               feed_dict=feed_dict,
                                               run_metadata=run_metadata)
            average_loss += loss_val

        # Add returned summaries to writer in each step.
            writer.add_summary(summary, step)
        # Add metadata to visualize the graph for the last run.
            if step == (num_steps - 1):
                writer.add_run_metadata(run_metadata, 'step%d' % step)

            if step % 20000 == 0:
                if step > 0:
                    average_loss /= 20000
                print('Average loss at step ', step, ': ', average_loss)
                average_loss = 0

        final_embeddings = normalized_embeddings.eval()

    writer.close()
    return final_embeddings


def get_std_mean(lab_value, get_mean=True, get_std=True, num_samples=1000):
    """
    To help assess the magnitude of cosine similarity when examining a specific pair,
    this function calculates the mean and standard deviation across a selection of
    randomly sampled pairs
    
    Args:
        lab_value: dictionary of words and their associated embedding vectors
        get_mean, get_std: flag to indicate which statistic should be returned
        num_samples: number of words in vocab to be randomly selected for use in creating an
                     nxn matrix of cosine similarity
    Returns:
        Depending on flags selected, mean and standard deviation of cosine similarity
        across all possible pairings of num_samples words
    """
    all_similarities = []
    keys1 = random.sample(list(lab_value), num_samples) # generate random words
    keys2 = random.sample(list(lab_value), num_samples)

    # Iterate through both  sets of words
    # This is a simple but inefficient way to do it, and ignores that cos_sim(key1, key2) = cos_sim(key2, key1)
    # It works well for small num_samples, but since the nested loop runs in O(n^2) it should be optimized if
    # large n is required
    for key1 in keys1: 
        for key2 in keys2:
            if key1 != key2:
            # find the cosine_similarity of the words
                all_similarities.append(cosine_similarity([lab_value[key1]],[lab_value[key2]]))

    std = np.std(all_similarities) # calculate std of all similarities
    mean = np.mean(all_similarities) # calculate mean of all similarities
    if get_mean:
        return mean
    if get_stg:
        return std
    

def fit_Reuters_LR_and_get_stats(final_embeddings):
    """Replaces text from Reuters database with the trained embeddings
       then fits a model using Logistic Regression, and checks to see
       how accurately the label can be predicted

       Function assumes that training data for Reuters is already in memory
       and that the previous functions in this module have all been run"""
    lab_all = [reverse_dictionary[i] for i in xrange(len(final_embeddings))]
    lab_value = dict(zip(lab_all, final_embeddings))

    vectorizer = MeanEmbeddingVectorizer(lab_value)
    feat_vect_train = vectorizer.transform(train_data)
    feat_vect_test = vectorizer.transform(test_data)
    log = LogisticRegression(penalty='l2',C=5)
    log.fit(feat_vect_train, train_labels)
    print("Accuracy with these hyperparamters: {0:.1f}%".format(metrics.accuracy_score(test_labels, log.predict(feat_vect_test))*100))
    print("man v woman", cosine_similarity([lab_value['man']], [lab_value['woman']]))
    print("king v queen", cosine_similarity([lab_value['king']], [lab_value['queen']]))
    

def create_cos_sim_matrix(row, col):
    flat_vals = [cosine_similarity([lab_value[a]],[lab_value[b]])[0][0] for a in row for b in col]
    matrix = np.array(flat_vals).reshape(len(row), len(col))
    df = pd.DataFrame(matrix, index=row, columns=col)
    return df


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([np.mean(
            [self.word2vec[word] for word in sents.split() if word in self.word2vec]
            or [np.zeros(self.dim)], axis=0) for sents in X])


class TfidfEmbeddingVectorizer(object):
    
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(next(iter(word2vec.values())))

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
    
if __name__ == '__main__':
    main()
