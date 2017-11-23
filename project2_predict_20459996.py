import tensorflow as tf
import numpy as np
from skimage import io, transform
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

width = height = 150
num_channels = 3


def read_test_data(file):
    pictures = []
    data = open(file)
    for line in data:
        line = line.replace('\n', '').split(' ')
        print("Reading file: "+line[0])
        img = io.imread(line[0])
        img = transform.resize(img, (width, height), mode='constant')
        pictures.append(img)
    np.asarray(pictures, np.float32)
    return pictures


if __name__ == '__main__':
    testList = read_test_data('./test.txt')
    labels = []
    #  Let us restore the saved model
    sess = tf.Session()
    # Step-1: Recreate the network graph. At this step only graph is created.
    saver = tf.train.import_meta_graph('./proj2_model2.meta')
    # Step-2: Now let's load the weights saved using the restore method.
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    i = 1
    for test in testList:
        # The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
        x_batch = test.reshape(1, width, height, num_channels)
        # Accessing the default graph which we have restored
        graph = tf.get_default_graph()

        # Now, let's get hold of the op that we can be processed to get the output.
        # In the original network y_pred is the tensor that is the prediction of the network
        y_pred = graph.get_tensor_by_name("y_pred:0")

        #  Let's feed the images to the input placeholders
        x = graph.get_tensor_by_name("x:0")
        y_true = graph.get_tensor_by_name("y_true:0")
        y_test_images = np.zeros((1, 5))

        # Creating the feed_dict that is required to be fed to calculate y_pred
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        predList = sess.run(y_pred, feed_dict=feed_dict_testing)
        predCls = sess.run(tf.argmax(predList, 1))
        labels.append(predCls[0])
        print("predicting -- ", i, " / ", len(testList))
        i += 1

    # save the predictions into a txt file
    with open('project2_20459996.txt', mode='wt', encoding='utf-8') as outfile:
        outfile.writelines("%d\n" % label for label in labels)
    sess.close()
