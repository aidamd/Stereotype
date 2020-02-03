from ntap.models import *

class Annotator(RNN):

    def build(self, data):
        RNN.build(self, data)
        self.vars["annotator"] = tf.placeholder(tf.int64, shape=[None],
                                                 name="Annotator")
        self.vars["gather"] = tf.placeholder(tf.int64, shape=[None, 2],
                                             name= "Gather")
        target = "hate"
        for annotator in data.annotators:
            n_outputs = 2
            logits = tf.layers.dense(self.vars["hidden_states"], n_outputs)
            weight = tf.gather(self.vars["weights-{}".format(target)],
                               self.vars["target-{}".format(target)])
            xentropy = tf.losses.sparse_softmax_cross_entropy\
                (labels=self.vars["target-{}".format(target)],
                    logits=logits, weights=weight)
            self.vars["loss-{}".format(annotator)] = tf.reduce_mean(xentropy)
            self.vars["predicted-{}".format(annotator)] = tf.argmax(logits, 1)
            self.vars["accuracy-{}".format(annotator)] = tf.reduce_mean(
                tf.cast(tf.equal(self.vars["predicted-{}".format(annotator)],
                                 self.vars["target-{}".format(target)]), tf.float32))

        self.vars["loss"] = tf.convert_to_tensor([
            self.vars["loss-{}".format(annotator)] for annotator in data.annotators],
            tf.float32)

        self.vars["accuracy"] = tf.convert_to_tensor(
            [self.vars["accuracy-{}".format(annotator)] for annotator in data.annotators],
            tf.float32)

        self.vars["prediction"] = tf.convert_to_tensor([
            self.vars["predicted-{}".format(annotator)] for annotator in
            data.annotators])

        self.vars["prediction-hate"] = tf.reshape(tf.gather_nd(self.vars["prediction"],
                                                           self.vars["gather"]), [-1])

        self.vars["joint_loss"] = tf.reduce_mean(tf.gather(self.vars["loss"],
                                                           self.vars["annotator"]))

        self.vars["joint_accuracy"] = tf.reduce_mean(tf.gather(self.vars["accuracy"],
                                                               self.vars["annotator"]))
        self.init = tf.global_variables_initializer()

class xAnnotator(RNN):

    def build(self, data):
        RNN.build(self, data)

        self.vars["annotators"] = tf.placeholder(tf.int64, shape=[None],
                                                 name="Annotators")

        self.vars["loss"] = tf.convert_to_tensor([self.vars["loss-" + name] for name in
                                         list(data.target_names.keys())], tf.float32)
        self.vars["accuracy"] = tf.convert_to_tensor([self.vars["accuracy-" + name] for name in
                                         list(data.target_names.keys())], tf.float32, )

        self.vars["joint_loss"] = tf.reduce_mean(tf.gather(self.vars["loss"],
                                                           self.vars["annotators"]))
        self.acc = tf.gather(self.vars["accuracy"], self.vars["annotators"])
        self.vars["joint_accuracy"] = tf.reduce_mean(self.acc)

    def evaluate(self, predictions, labels, num_classes,
                 metrics=["f1", "accuracy", "precision", "recall", "kappa"]):
        stats = list()
        all_y, all_y_hat = list(), list()
        for key in predictions:
            target_key = key.replace("prediction-", "target-")
            if not key.startswith("prediction-"):
                continue
            if target_key not in labels:
                raise ValueError("Predictions and Labels have different keys")
            stat = {"Target": key.replace("prediction-", "")}
            y, y_hat = labels[target_key], predictions[key]
            idx = [i for i in range(len(y)) if y[i] != 2]
            sub_y, sub_y_hat = [lab for i, lab in enumerate(y) if i in idx], \
                               [lab for i, lab in enumerate(y_hat) if i in idx]
            for i in range(len(sub_y)):
                if y_hat[i] == 2:
                    y_hat[i] = 1 - y[i]
            all_y.extend(sub_y); all_y_hat.extend(sub_y_hat)
            card = 2
        for m in metrics:
            if m == 'accuracy':
                stat[m] = accuracy_score(y, y_hat)
            avg = 'binary' if card == 2 else 'macro'
            if m == 'precision':
                stat[m] = precision_score(y, y_hat, average=avg)
            if m == 'recall':
                stat[m] = recall_score(y, y_hat, average=avg)
            if m == 'f1':
                stat[m] = f1_score(y, y_hat, average=avg)
            if m == 'kappa':
                stat[m] = cohen_kappa_score(y, y_hat)
        stats.append(stat)
        return stats

class AnnotatorDemo(RNN):


    def build(self, data):
        RNN.build(self, data)
        self.vars["annotator"] = tf.placeholder(tf.int32, shape=[None], name="Annotator")

        self.vars["demo"] = tf.placeholder(tf.int32, shape=[None, None], name="Demo")
        self.vars["DemoEmbeddingPlaceholder"] = tf.placeholder(tf.float32,
                                                               shape=[max(data.annotators) + 1,
                                                                      data.demo_dim])

        self.vars["Demo_Embedding"] = tf.nn.embedding_lookup(self.vars["DemoEmbeddingPlaceholder"],
                                                             self.vars["annotator"])

        self.vars["hidden_demo"] = tf.concat([self.vars["hidden_states"],
                                            self.vars["Demo_Embedding"]], axis=-1)

        for target in data.targets:
            n_outputs = 2
            logits = tf.layers.dense(tf.layers.dropout(self.vars["hidden_demo"],
                                                       rate = self.vars["keep_ratio"]),
                                                       n_outputs)
            weight = tf.gather(self.vars["weights-{}".format(target)],
                               self.vars["target-{}".format(target)])
            xentropy = tf.losses.sparse_softmax_cross_entropy \
                (labels=self.vars["target-{}".format(target)],
                 logits=logits, weights=weight)
            self.vars["loss-{}".format(target)] = tf.reduce_mean(xentropy)
            self.vars["prediction-{}".format(target)] = tf.argmax(logits, 1)
            self.vars["accuracy-{}".format(target)] = tf.reduce_mean(
                tf.cast(tf.equal(self.vars["prediction-{}".format(target)],
                                 self.vars["target-{}".format(target)]), tf.float32))
        self.init = tf.global_variables_initializer()

        """
        Annotator.build(self, data)
        for target in data.targets:
            n_outputs = len(data.target_names[target])
            self.vars["demo-{}".format(target)] = tf.placeholder(tf.int64,
                    shape=[None], name="demo-{}".format(target))
            tile_demo = tf.tile(tf.expand_dims(self.vars["demo-{}".format(target)], 0),
                                [tf.shape(self.vars["hidden_states"])[0], 1])
            self.vars["hidden-{}".format(target)] = tf.concat(self.vars["hidden_states"],
                                                              tile_demo)
            logits = tf.layers.dense("hidden-{}".format(target), n_outputs)
            weight = tf.gather(self.vars["weights-{}".format(target)],
                               self.vars["target-{}".format(target)])
            xentropy = tf.losses.sparse_softmax_cross_entropy \
                (labels=self.vars["target-{}".format(target)],
                 logits=logits, weights=weight)
            self.vars["loss-{}".format(target)] = tf.reduce_mean(xentropy)
            self.vars["prediction-{}".format(target)] = tf.argmax(logits, 1)
            self.vars["accuracy-{}".format(target)] = tf.reduce_mean(
                tf.cast(tf.equal(self.vars["prediction-{}".format(target)],
                                 self.vars["target-{}".format(target)]), tf.float32))
        """

class xAnnotatorDemo(RNN):

    def build(self, data):
        RNN.build(self, data)
        self.vars["annotators"] = tf.placeholder(tf.int64, shape=[None],
                                                 name="Annotators")
        self.vars["DemoEmbeddingPlaceholder"] = tf.placeholder(tf.float32,
                                                               shape=[max(data.annotators) + 1,
                                                                      data.demo_dim])

        for target in data.targets:
            n_outputs = len(data.target_names[target])
            self.anno_id = tf.tile(tf.convert_to_tensor([int(target)]),
                              [tf.shape(self.vars["hidden_states"])[0]])

            self.vars["annotator-demo"] = tf.nn.embedding_lookup(self.vars["DemoEmbeddingPlaceholder"],
                                                              self.anno_id)

            self.vars["hidden-{}".format(target)] = tf.layers.dropout(tf.concat([self.vars["hidden_states"],
                                                  self.vars["annotator-demo"]], axis=-1),
                                                                      rate = self.vars["keep_ratio"])

            logits = tf.layers.dense(self.vars["hidden-{}".format(target)], n_outputs)
            weight = tf.gather(self.vars["weights-{}".format(target)],
                               self.vars["target-{}".format(target)])
            xentropy = tf.losses.sparse_softmax_cross_entropy \
                (labels=self.vars["target-{}".format(target)],
                 logits=logits, weights=weight)
            self.vars["loss-{}".format(target)] = tf.reduce_mean(xentropy)
            self.vars["prediction-{}".format(target)] = tf.argmax(logits, 1)
            self.vars["accuracy-{}".format(target)] = tf.reduce_mean(
                tf.cast(tf.equal(self.vars["prediction-{}".format(target)],
                                 self.vars["target-{}".format(target)]), tf.float32))

        self.vars["loss"] = tf.convert_to_tensor([self.vars["loss-" + name] for name in
                                                  list(data.target_names.keys())], tf.float32)
        self.vars["accuracy"] = tf.convert_to_tensor([self.vars["accuracy-" + name] for name in
                                                      list(data.target_names.keys())], tf.float32, )

        #self.vars["joint_loss"] = tf.reduce_mean(tf.gather(self.vars["loss"],
        #                                                   self.vars["annotators"]))
        self.vars["joint_loss"] = tf.reduce_mean(self.vars["loss"])
        self.acc = tf.gather(self.vars["accuracy"], self.vars["annotators"])
        self.vars["joint_accuracy"] = tf.reduce_mean(self.acc)
        self.init = tf.global_variables_initializer()

    def evaluate(self, predictions, labels, num_classes,
                 metrics=["f1", "accuracy", "precision", "recall", "kappa"]):
        stats = list()
        all_y, all_y_hat = list(), list()
        for key in predictions:
            target_key = key.replace("prediction-", "target-")
            if not key.startswith("prediction-"):
                continue
            if target_key not in labels:
                raise ValueError("Predictions and Labels have different keys")
            stat = {"Target": key.replace("prediction-", "")}
            y, y_hat = labels[target_key], predictions[key]
            idx = [i for i in range(len(y)) if y[i] != 2]
            sub_y, sub_y_hat = [lab for i, lab in enumerate(y) if i in idx], \
                               [lab for i, lab in enumerate(y_hat) if i in idx]
            all_y.extend(sub_y); all_y_hat.extend(sub_y_hat)
            card = num_classes[key]
        for m in metrics:
            if m == 'accuracy':
                stat[m] = accuracy_score(y, y_hat)
            avg = 'binary' if card == 2 else 'macro'
            if m == 'precision':
                stat[m] = precision_score(y, y_hat, average=avg)
            if m == 'recall':
                stat[m] = recall_score(y, y_hat, average=avg)
            if m == 'f1':
                stat[m] = f1_score(y, y_hat, average=avg)
            if m == 'kappa':
                stat[m] = cohen_kappa_score(y, y_hat)
        stats.append(stat)
        return stats