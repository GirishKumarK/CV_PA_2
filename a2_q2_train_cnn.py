import time


import tensorflow as tf

########### Convolutional neural network class ############
class ConvNet(object):
    def __init__(self, mode):
        self.mode = mode

    # Read train, valid and test data.
    def read_data(self, train_set, test_set):
        # Load train set.
        trainX = train_set.images
        trainY = train_set.labels

        # Load test set.
        testX = test_set.images
        testY = test_set.labels

        return trainX, trainY, testX, testY

    # Baseline model. step 1
    def model_1(self, X, hidden_size):
        # ======================================================================
        # One fully connected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        pool_flat = tf.reshape(X, [-1, 28 * 28 * 1])
        dense = tf.layers.dense(inputs=pool_flat, units=hidden_size, activation=tf.nn.sigmoid)
        fcl = dense
        #
        # Uncomment the following return stmt once method implementation is done.
        return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        # return NotImplementedError()

    # Use two convolutional layers.
    def model_2(self, X, hidden_size):
        # ======================================================================
        # Two convolutional layers + one fully connnected layer.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        conv1 = tf.layers.conv2d(inputs=X, filters=40, kernel_size=[5, 5], padding="same", activation=tf.nn.sigmoid)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=40, kernel_size=[5, 5], padding="same", activation=tf.nn.sigmoid)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=1)
        # pool2_flat = tf.reshape(pool2, [10, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2, units=hidden_size, activation=tf.nn.sigmoid)
        fcl = dense
        #
        # Uncomment the following return stmt once method implementation is done.
        return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        # return NotImplementedError()

    # Replace sigmoid with ReLU.
    def model_3(self, X, hidden_size):
        # ======================================================================
        # Two convolutional layers + one fully connected layer, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        conv1 = tf.layers.conv2d(inputs=X, filters=40, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=40, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=1)
        # pool2_flat = tf.reshape(pool2, [10, 7 * 7 * 64])
        dense = tf.layers.dense(inputs=pool2, units=hidden_size, activation=tf.nn.relu)
        fcl = dense
        #
        # Uncomment the following return stmt once method implementation is done.
        return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        # return NotImplementedError()

    # Add one extra fully connected layer.
    def model_4(self, X, hidden_size, decay):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        regularizer = tf.contrib.layers.l2_regularizer(scale=decay)
        conv1 = tf.layers.conv2d(inputs=X, filters=40, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=40, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=1)
        # pool2_flat = tf.reshape(pool2, [10, 7 * 7 * 64])
        dense1 = tf.layers.dense(inputs=pool2, units=hidden_size, activation=tf.nn.relu, kernel_regularizer=regularizer)
        # pool2_flat = tf.reshape(pool2, [10, 7 * 7 * 64])
        dense2 = tf.layers.dense(inputs=dense1, units=hidden_size, activation=tf.nn.relu, kernel_regularizer=regularizer)
        fcl = dense2
        #
        # Uncomment the following return stmt once method implementation is done.
        return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        # return NotImplementedError()

    # Use Dropout now.
    def model_5(self, X, hidden_size, is_train):
        # ======================================================================
        # Two convolutional layers + two fully connected layers, with ReLU.
        # and  + Dropout.
        #
        # ----------------- YOUR CODE HERE ----------------------
        #
        conv1 = tf.layers.conv2d(inputs=X, filters=40, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=1)
        conv2 = tf.layers.conv2d(inputs=pool1, filters=40, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=1)
        # pool2_flat = tf.reshape(pool2, [10, 7 * 7 * 64])
        dense1 = tf.layers.dense(inputs=pool2, units=hidden_size, activation=tf.nn.relu)
        # pool2_flat = tf.reshape(pool2, [10, 7 * 7 * 64])
        dense2 = tf.layers.dense(inputs=dense1, units=hidden_size, activation=tf.nn.relu)
        dropout = tf.layers.dropout(inputs=dense2, rate=0.5, training=is_train)
        fcl = dropout
        #
        # Uncomment the following return stmt once method implementation is done.
        return  fcl
        # Delete line return NotImplementedError() once method is implemented.
        # return NotImplementedError()

    # Entry point for training and evaluation.
    def train_and_evaluate(self, FLAGS, train_set, test_set):
        class_num = 10
        num_epochs = FLAGS.num_epochs
        batch_size = FLAGS.batch_size
        learning_rate = FLAGS.learning_rate
        hidden_size = FLAGS.hiddenSize
        decay = FLAGS.decay

        trainX, trainY, testX, testY = self.read_data(train_set, test_set)

        input_size = trainX.shape[1]
        train_size = trainX.shape[0]
        test_size = testX.shape[0]

        trainX = trainX.reshape((-1, 28, 28, 1))
        testX = testX.reshape((-1, 28, 28, 1))

        with tf.Graph().as_default():
            # Input data
            X = tf.placeholder(tf.float32, [None, 28, 28, 1])
            Y = tf.placeholder(tf.float32, [None, class_num])
            is_train = tf.placeholder(tf.bool)

            # model 1: base line
            if self.mode == 1:
                features = self.model_1(X, hidden_size)

            # model 2: use two convolutional layer
            elif self.mode == 2:
                features = self.model_2(X, hidden_size)

            # model 3: replace sigmoid with relu
            elif self.mode == 3:
                features = self.model_3(X, hidden_size)


            # model 4: add one extral fully connected layer
            elif self.mode == 4:
                features = self.model_4(X, hidden_size, decay)

            # model 5: utilize dropout
            elif self.mode == 5:
                features = self.model_5(X, hidden_size, is_train)

            # ======================================================================
            # Define softmax layer, use the features.
            # ----------------- YOUR CODE HERE ----------------------
            #
            w = tf.Variable(tf.truncated_normal([hidden_size, class_num]))
            b = tf.Variable(tf.zeros([class_num]))
            logits = tf.matmul(features, w) + b
            logits = tf.nn.softmax(logits)
            #
            # Remove NotImplementedError and assign calculated value to logits after code implementation.
            # logits = NotImplementedError

            # ======================================================================
            # Define loss function, use the logits.
            # ----------------- YOUR CODE HERE ----------------------
            #
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)
            loss = tf.reduce_mean(cross_entropy)
            #onehot_labels = tf.one_hot(indices=tf.cast(labels=Y, tf.int32), depth=10)
            #loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)
            #
            # Remove NotImplementedError and assign calculated value to loss after code implementation.
            # loss = NotImplementedError

            # ======================================================================
            # Define training op, use the loss.
            # ----------------- YOUR CODE HERE ----------------------
            #
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            train_op = optimizer.minimize(loss=loss)
            #
            # Remove NotImplementedError and assign calculated value to train_op after code implementation.
            # train_op = NotImplementedError

            # ======================================================================
            # Define accuracy op.
            # ----------------- YOUR CODE HERE ----------------------
            #
            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
            correct_prediction = tf.cast(correct_prediction, tf.float32)
            accuracy = tf.reduce_mean(correct_prediction)
            #
            # accuracy = NotImplementedError

            # ======================================================================
            # Allocate percentage of GPU memory to the session.
            # If you system does not have GPU, set has_GPU = False
            #
            has_GPU = True
            if has_GPU:
                gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction=1.0, 
                                           allocator_type='BFC', 
                                           deferred_deletion_bytes=0, 
                                           allow_growth=True)
                config = tf.ConfigProto(gpu_options=gpu_option)
            else:
                config = tf.ConfigProto()

            # Create TensorFlow session with GPU setting.
            with tf.Session(config=config) as sess:
                tf.global_variables_initializer().run()

                for i in range(num_epochs):
                    print(20 * '*', 'epoch', i + 1, 20 * '*')
                    start_time = time.time()
                    s = 0
                    while s < train_size:
                        e = min(s + batch_size, train_size)
                        batch_x = trainX[s: e]
                        batch_y = trainY[s: e]
                        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, is_train: True})
                        s = e
                    end_time = time.time()
                    print ('the training took: %d(s)' % (end_time - start_time))

                    #total_correct = sess.run(accuracy, feed_dict={X: testX, Y: testY, is_train: False})
                    #print ('accuracy of the trained model %f' % (total_correct / testX.shape[0]))
                    total_correct = sess.run(accuracy, feed_dict={X: batch_x, Y: batch_y, is_train: False})
                    print ('accuracy of the trained model %f' % (total_correct))
                    print ()

                #return sess.run(accuracy, feed_dict={X: testX, Y: testY, is_train: False}) / testX.shape[0]
                return sess.run(accuracy, feed_dict={X: testX, Y: testY, is_train: False})

