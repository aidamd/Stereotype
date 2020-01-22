from ntap.models import *


class Annotator(RNN):

    def build(self, data):
        RNN.build(self, data)

        self.vars["annotators"] = tf.placeholder(tf.int32, shape=[None],
                                                 name="Annotators")

        self.vars["loss"] = tf.convert_to_tensor([self.vars["loss-" + name] for name in
                                         list(data.target_names.keys())], tf.float32)
        self.vars["accuracy"] = tf.convert_to_tensor([self.vars["accuracy-" + name] for name in
                                         list(data.target_names.keys())], tf.float32, )

        self.vars["joint_loss"] = tf.reduce_mean(tf.gather(self.vars["loss"],
                                                           self.vars["annotators"]))
        self.vars["joint_accuracy"] = tf.reduce_mean(tf.gather(self.vars["accuracy"],
                                                               self.vars["annotators"]))

    def evaluate(self, predictions, labels, num_classes,
                 metrics=["f1", "accuracy", "precision", "recall", "kappa"]):
        stats = list()
        for key in predictions:
            if not key.startswith("prediction-"):
                continue
            if key not in labels:
                raise ValueError("Predictions and Labels have different keys")
            stat = {"Target": key.replace("prediction-", "")}
            y, y_hat = labels[key], predictions[key]
            idx = [i for i in range(y.size) if y[i] != 2]
            y, y_hat = np.take(y, idx), np.take(y_hat, idx)
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

class AnnotatorInfo(RNN):


    def build(self):

