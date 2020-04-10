from ntap.models import *

class Annotator(RNN):

    def build(self, data):
        RNN.build(self, data)
        self.vars["annotator"] = tf.placeholder(tf.int64, name="Annotator")
        #self.vars["gather"] = tf.placeholder(tf.int64, shape=[None, 2],
        #                                     name= "Gather")
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

        self.vars["prediction-hate"] = tf.gather(self.vars["prediction"], self.vars["annotator"])

        self.vars["joint_loss"] = tf.gather(self.vars["loss"], self.vars["annotator"])

        self.vars["joint_accuracy"] = tf.gather(self.vars["accuracy"],
                                                self.vars["annotator"])
        if self.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'momentum':
            opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            raise ValueError("Invalid optimizer specified")
        self.vars["training_op"] = opt.minimize(loss=self.vars["joint_loss"])
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


class MultiModel(RNN):

    def build(self, data):
        tf.reset_default_graph()
        self.demo_size = 64
        n_outputs = 2
        self.vars["sequence_length"] = tf.placeholder(tf.int32, shape=[None],
                name="SequenceLength")
        self.vars["word_inputs"] = tf.placeholder(tf.int32, shape=[None, None],
                                                  name="RNNInput")
        self.vars["keep_ratio"] = tf.placeholder(tf.float32, name="KeepRatio")
        W = tf.Variable(tf.constant(0.0, shape=[len(data.vocab), data.embed_dim]),
                        trainable=False, name="Embed")
        self.vars["Embedding"] = tf.nn.embedding_lookup(W,
                self.vars["word_inputs"])
        self.vars["EmbeddingPlaceholder"] = tf.placeholder(tf.float32,
                shape=[len(data.vocab), data.embed_dim])
        self.vars["EmbeddingInit"] = W.assign(self.vars["EmbeddingPlaceholder"])
        self.vars["states"] = self._RNN__build_rnn(self.vars["Embedding"],
                self.hidden_size, self.cell_type, self.bi,
                self.vars["sequence_length"])

        if self.rnn_dropout is not None:
            self.vars["hidden_states"] = \
                tf.layers.dropout(self.vars["states"],
                                  rate=self.vars["keep_ratio"],
                                  name="RNNDropout")
        else:
            self.vars["hidden_states"] = self.vars["states"]

        if data.demo is not None:
            self.vars["DemoEmbeddingPlaceholder"] = \
                tf.placeholder(tf.float32, shape=[max(data.annotators) + 1, data.demo_dim],
                               name="DemoEmbedding")
            We = tf.Variable(tf.random_uniform(shape=
                [2 * self.hidden_size + self.demo_size, n_outputs]))
            b = tf.Variable(tf.zeros([n_outputs]))

        for target in data.targets:
            if data.demo is not None:
                self.anno_demo = tf.nn.embedding_lookup(
                    self.vars["DemoEmbeddingPlaceholder"],
                    tf.convert_to_tensor([int(target)]))

                self.vars["annotator-demo"] = tf.tile(
                    tf.layers.dense(self.anno_demo, self.demo_size, activation=tf.nn.relu),
                    [tf.shape(self.vars["hidden_states"])[0], 1])

                self.vars["hidden-{}".format(target)] = \
                    tf.concat([self.vars["hidden_states"],
                               self.vars["annotator-demo"]], axis=-1)
                logits = tf.matmul(self.vars["hidden-{}".format(target)], We) + b
                #logits = tf.layers.dense(self.vars["hidden-{}".format(target)], n_outputs)

            else:
                self.vars["hidden-{}".format(target)] = self.vars["hidden_states"]
                logits = tf.layers.dense(self.vars["hidden-{}".format(target)], n_outputs)

            self.vars["target-{}".format(target)] = \
                tf.placeholder(tf.int64, shape=[None],
                               name="target-{}".format(target))

            self.vars["weights-{}".format(target)] = \
                tf.placeholder(tf.float32, shape=[n_outputs],
                               name="weights-{}".format(target))

            self.vars["mask-{}".format(target)] = \
                tf.placeholder(tf.bool, shape=[None],
                               name="mask-{}".format(target))

            self.labels = tf.boolean_mask(self.vars["target-{}".format(target)],
                                         self.vars["mask-{}".format(target)])
            weight = tf.gather(self.vars["weights-{}".format(target)],
                               self.labels)
            self.logits = tf.boolean_mask(logits, tf.cast(self.vars["mask-{}".format(target)],
                                                 tf.float32))
            xentropy = tf.losses.sparse_softmax_cross_entropy \
                (labels=self.labels,
                 logits=self.logits,
                 weights=weight)
            self.vars["loss-{}".format(target)] = tf.reduce_mean(xentropy)
            self.vars["prediction-{}".format(target)] = tf.argmax(logits, 1)
            self.prediction = tf.boolean_mask(self.vars["prediction-{}".format(target)],
                                          self.vars["mask-{}".format(target)])
            self.vars["accuracy-{}".format(target)] = tf.reduce_mean(
                tf.cast(tf.equal(self.prediction,
                                 self.labels), tf.float32))
        self.vars["joint_loss"] = \
            sum([self.vars[name] for name in self.vars if name.startswith("loss")])

        self.vars["joint_accuracy"] = \
            sum([self.vars[name] for name in self.vars if name.startswith("accuracy")]) \
            / len([self.vars[name] for name in self.vars if name.startswith("accuracy")])

        if self.optimizer == 'adam':
            opt = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'adagrad':
            opt = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'momentum':
            opt = tf.train.MomentumOptimizer(learning_rate=self.learning_rate)
        elif self.optimizer == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
        else:
            raise ValueError("Invalid optimizer specified")
        self.vars["training_op"] = opt.minimize(loss=self.vars["joint_loss"])
        self.init = tf.global_variables_initializer()

    def evaluate(self, predictions, labels, num_classes,
                 metrics=["f1", "accuracy", "precision", "recall", "kappa"]):
        stats = list()
        all_y, all_y_hat = list(), list()
        for key in predictions:
            #target_key = key.replace("prediction-", "target-")
            if not key.startswith("prediction-"):
                continue
            if key not in labels:
                raise ValueError("Predictions and Labels have different keys")
            stat = {"Target": key.replace("prediction-", "")}
            y, y_hat = labels[key], predictions[key]
            idx = [i for i in range(len(y)) if y[i] != -1]
            sub_y, sub_y_hat = [lab for i, lab in enumerate(y) if i in idx], \
                               [lab for i, lab in enumerate(y_hat) if i in idx]
            all_y.extend(sub_y); all_y_hat.extend(sub_y_hat)
            card = num_classes[key]
            for m in metrics:
                if m == 'accuracy':
                    stat[m] = accuracy_score(sub_y, sub_y_hat)
                avg = 'binary' if card == 2 else 'macro'
                if m == 'precision':
                    stat[m] = precision_score(sub_y, sub_y_hat, average=avg)
                if m == 'recall':
                    stat[m] = recall_score(sub_y, sub_y_hat, average=avg)
                if m == 'f1':
                    stat[m] = f1_score(sub_y, sub_y_hat, average=avg)
                if m == 'kappa':
                    stat[m] = cohen_kappa_score(sub_y, sub_y_hat)
            stats.append(stat)
        stat = {"Target": "all"}
        for m in metrics:
            if m == 'accuracy':
                stat[m] = accuracy_score(all_y, all_y_hat)
            avg = 'binary' if card == 2 else 'macro'
            if m == 'precision':
                stat[m] = precision_score(all_y, all_y_hat, average=avg)
            if m == 'recall':
                stat[m] = recall_score(all_y, all_y_hat, average=avg)
            if m == 'f1':
                stat[m] = f1_score(all_y, all_y_hat, average=avg)
            if m == 'kappa':
                stat[m] = cohen_kappa_score(all_y, all_y_hat)
        stats.append(stat)
        return stats
