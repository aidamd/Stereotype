from ntap.models import *


class Multi(RNN):
    def loss(self, tasks):
        joint_loss = sum([self.vars[name] for name in self.vars if name.startswith("loss") and
                          name[5:] in tasks])
        joint_accuracy = sum([self.vars[name] for name in self.vars if name.startswith("accuracy") and
                          name[9:] in tasks]) / tf.shape(tasks)[0]
        return joint_loss, joint_accuracy

    def build(self, data):
        RNN.build(self, data)
        #self.vars["annotators"] = tf.placeholder(tf.string, shape=[None],
        #                                          name="Annotators")
        #self.vars["joint_loss"], self.vars["joint_accuracy"] = \
        #    self.loss(self.vars["annotators"])

    def train(self, data, num_epochs=30, batch_size=256, train_indices=None,
              test_indices=None, model_path=None):
        saver = tf.train.Saver()
        with tf.Session() as self.sess:
            self.sess.run(self.init)
            _ = self.sess.run(self.vars["EmbeddingInit"],
                feed_dict={self.vars["EmbeddingPlaceholder"]: data.embedding})
            for epoch in range(num_epochs):
                epoch_loss, train_accuracy, test_accuracy = 0.0, 0.0, 0.0
                num_batches, test_batches = 0, 0
                for i, feed in enumerate(data.high_batches(self.vars,
                    batch_size, test=False, keep_ratio=self.rnn_dropout,
                    idx=train_indices)):
                    _, loss_val, acc = self.sess.run([self.vars["training_op"],
                        self.vars["joint_loss"], self.vars["joint_accuracy"]],
                                                     feed_dict=feed)
                    epoch_loss += loss_val
                    train_accuracy += acc
                    num_batches += 1
                for i, feed in enumerate(data.high_batches(self.vars,
                    batch_size, test=False, keep_ratio=self.rnn_dropout,
                    idx=test_indices)):
                    acc = self.sess.run(self.vars["joint_accuracy"], feed_dict=feed)
                    test_accuracy += acc
                    test_batches += 1

                print("Epoch {}: Loss = {:.3}, Train Accuracy = {:.3}, Test Accuracy = {:.3}"
                      .format(epoch, epoch_loss/num_batches, train_accuracy/num_batches,
                              test_accuracy/test_batches))
            if model_path is not None:
                saver.save(self.sess, model_path)
        return