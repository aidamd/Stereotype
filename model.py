from ntap.models import *


class Multi(RNN):

    def build(self, data):
        RNN.build(self, data)
        self.vars["annotators"] = tf.placeholder(tf.int32, shape=[None],
                                                 name="Annotators")
        self.vars["loss"] = tf.convert_to_tensor([self.vars["loss-" + name] for name in
                                         list(data.target_names.keys())], tf.float32)
        self.vars["accuracy"] = tf.convert_to_tensor([self.vars["accuracy-" + name] for name in
                                         list(data.target_names.keys())], tf.float32, )
        self.vars["joint_loss"] = tf.reduce_sum(tf.gather(self.vars["loss"], self.vars["annotators"]))
        self.vars["joint_accuracy"] = tf.reduce_mean(tf.gather(self.vars["accuracy"], self.vars["annotators"]))

        #self.vars["joint_loss"], self.vars["joint_accuracy"] = \
        #    self.loss(self.vars["annotators"])
