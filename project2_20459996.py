import tensorflow as tf
import numpy as np
from skimage import io, transform
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

width = height = 150
num_channels = 3
num_classes = 5
batch_size = 64
total_epochs = 40
drop_out = 0.50
x = tf.placeholder(tf.float32, shape=[None, width, height, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, 1)

# Network graph params
filter_size_conv1 = 3
num_filters_conv1 = 32
filter_size_conv2 = 3
num_filters_conv2 = 64
filter_size_conv3 = 3
num_filters_conv3 = 64
fc_layer_size = 128


def read_data(file):
    pictures = []
    label = []
    data = open(file)
    for line in data:
        line = line.replace('\n', '').split(' ')
        print("Reading file: "+line[0])
        img = io.imread(line[0])
        img = transform.resize(img, (width, height), mode='constant')
        pictures.append(img)
        label.append(int(line[1]))
    np.asarray(pictures, np.float32)
    np.asarray(label, np.int32)
    return pictures, label


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))


def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    biases = create_biases(num_filters)
    # Creating the convolutional layer
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
    layer += biases
    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    layer = tf.nn.relu(layer)
    return layer


def create_flatten_layer(layer):
    # Get the shape of the layer from the previous layer.
    layer_shape = layer.get_shape()
    # Number of features will be img_height * img_width* num_channels.
    num_features = layer_shape[1:4].num_elements()
    # Flatten the layer
    layer = tf.reshape(layer, [-1, num_features])
    return layer


def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
    # Fully connected layer takes input x and produces wx+b.
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    return layer


layer_conv1 = create_convolutional_layer(input=x,
                                         num_input_channels=num_channels,
                                         conv_filter_size=filter_size_conv1,
                                         num_filters=num_filters_conv1)
# Apply Dropout
layer_conv1 = tf.nn.dropout(layer_conv1, drop_out)

layer_conv2 = create_convolutional_layer(input=layer_conv1,
                                         num_input_channels=num_filters_conv1,
                                         conv_filter_size=filter_size_conv2,
                                         num_filters=num_filters_conv2)
# Apply Dropout
layer_conv2 = tf.nn.dropout(layer_conv2, drop_out)

layer_conv3 = create_convolutional_layer(input=layer_conv2,
                                         num_input_channels=num_filters_conv2,
                                         conv_filter_size=filter_size_conv3,
                                         num_filters=num_filters_conv3)
# Apply Dropout
layer_conv3 = tf.nn.dropout(layer_conv3, drop_out)

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                            num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                            num_outputs=fc_layer_size,
                            use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                            num_inputs=fc_layer_size,
                            num_outputs=num_classes,
                            use_relu=False)

y_pred = tf.nn.softmax(layer_fc2, name='y_pred')

y_pred_cls = tf.argmax(y_pred, 1)


session = tf.Session()

session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
session.run(tf.global_variables_initializer())

if __name__ == '__main__':
    train, trainLabel0 = read_data('./train.txt')
    val, valLabel0 = read_data('./val.txt')

    trainLabel = []
    valLabel = []
    for i in trainLabel0:
        newLabel = np.zeros(num_classes)
        newLabel[i] = 1.0
        trainLabel.append(newLabel)
    trainLabel = np.array(trainLabel)
    for i in valLabel0:
        newLabel = np.zeros(num_classes)
        newLabel[i] = 1.0
        valLabel.append(newLabel)
    valLabel = np.array(valLabel)

    total_batches_tr = int(len(trainLabel) / batch_size) + 1
    total_batches_val = int(len(valLabel) / batch_size) + 1

    for epoch in range(total_epochs):
        acc_tr = 0
        acc_val = 0
        loss_val = 0
        for batch in range(total_batches_tr):
            train_batch = train[batch*batch_size:(batch+1)*batch_size]
            trainLabel_batch = trainLabel[batch*batch_size:(batch+1)*batch_size]

            feed_tr = {x: train_batch, y_true: trainLabel_batch}

            session.run(optimizer, feed_dict=feed_tr)
            acc = session.run(accuracy, feed_dict=feed_tr)
            acc_tr += acc
        for batch in range(total_batches_val):
            val_batch = val[batch*batch_size:(batch+1)*batch_size]
            valLabel_batch = valLabel[batch*batch_size:(batch+1)*batch_size]

            feed_val = {x: val_batch, y_true: valLabel_batch}

            loss = session.run(cost, feed_dict=feed_val)
            acc = session.run(accuracy, feed_dict=feed_val)
            acc_val += acc
            loss_val += loss
        acc_tr_avg = acc_tr / total_batches_tr
        acc_val_avg = acc_val / total_batches_val
        loss_val_avg = loss_val / total_batches_tr
        msg = "Training Epoch {0}---" \
              "Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
        print(msg.format(epoch + 1, acc_tr_avg, acc_val_avg, loss_val_avg))

    saver = tf.train.Saver()
    saver.save(session, './proj2_model2')
    session.close()






