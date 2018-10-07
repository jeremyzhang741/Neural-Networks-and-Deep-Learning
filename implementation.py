import tensorflow as tf
import re
import string

BATCH_SIZE = 130
MAX_WORDS_IN_REVIEW = 150  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector
'''
stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})
'''
stop_words = set({'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
        'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
        'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they',
        'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom',
        'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did',
        'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as',
        'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
        'between', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
        'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
        'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just',
        'don', 'should', 'now','mr','miss','br','ill','re','would',
        've','ll','d','m','o','y','ain','aren','couldn','didn','doesn','hadn','hasn',
        'haven','isn','ma','mightn','mustn','needn','shan','shouldn','wasn','weren',
        'won','wouldn'})

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    review=review.lower()
    
    review = re.sub(r'[^\w\s]', "", review)
    collection=[]
    for x in review.split(' '):
        if x not in stop_words:
            collection.append(x)
    #processed_review=' '.join(x for x in collection)
    processed_review=collection
    return processed_review



def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """
    
    num_cells = 65
    num_class = 2
    input_data = tf.placeholder(tf.float32,[None, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE],name="input_data")
    labels = tf.placeholder(tf.float32,[None,num_class], name="labels")
    dropout_keep_prob = tf.placeholder_with_default(0.6,shape=())
    lstm_cell_1 = tf.contrib.rnn.GRUCell(num_cells)
    lstm_cell_2 = tf.contrib.rnn.GRUCell(num_cells)
    lstm_cell_1 = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_1, output_keep_prob=dropout_keep_prob)
    lstm_cell_2 = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_2, output_keep_prob=dropout_keep_prob)
    (value_1,value_2),_ = tf.nn.bidirectional_dynamic_rnn(cell_fw =lstm_cell_1, cell_bw = lstm_cell_2, dtype = tf.float32, inputs = input_data)
    final_value = tf.concat((value_1, value_2),2)
    final_output = final_value[:,-1,:]
    logits = tf.layers.dense(final_output,2)
    pred = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
    Accuracy = tf.reduce_mean(tf.cast(pred, tf.float32), name = 'accuracy')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels), name = 'loss')
    optimizer = tf.train.AdamOptimizer(learning_rate = 0.0001).minimize(loss)
    return input_data, labels, dropout_keep_prob, optimizer, Accuracy, loss
