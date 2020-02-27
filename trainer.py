import numpy as np
import tensorflow as tf
import msgnet

class DensityOutputTrainer(msgnet.train.Trainer):
    def setup_input_symbols(self):
        input_symbols = self.model.get_input_symbols()
        output_size = 1

        self.sym_graph_targets = tf.placeholder(
            tf.float32, shape=(None, None), name="sym_graph_targets"
        )

        input_symbols.update({"probes_target": self.sym_graph_targets})

        return input_symbols

    def setup_metrics(self):
        graph_error = self.model.get_graph_out() - self.input_symbols["probes_target"]

        metric_tensors = {"graph_error": graph_error}
        return metric_tensors

    def setup_total_cost(self):
        sym_graph_targets = self.input_symbols["probes_target"]
        graph_cost = self.get_cost_graph_target(sym_graph_targets, self.model)
        total_cost = graph_cost
        return total_cost

    def evaluate_metrics(self, session, datahandler, prefix="", decimation=1):
        target_mae = 0
        target_mse = 0
        num_vals = 0
        for input_data in datahandler.get_test_batches(decimation=decimation):
            feed_dict = {}
            for key in self.input_symbols.keys():
                feed_dict[self.input_symbols[key]] = input_data[key]
            syms = [self.metric_tensors["graph_error"]]
            graph_error, = session.run(syms, feed_dict=feed_dict)
            target_mae += np.sum(np.abs(graph_error))
            target_mse += np.sum(np.square(graph_error))
            num_vals += graph_error.shape[0]*graph_error.shape[1]

        if prefix:
            prefix += "_"
        metrics = {
            prefix + "mae": target_mae / num_vals,
            prefix + "rmse": np.sqrt(target_mse / num_vals),
        }

        return metrics

    @staticmethod
    def get_cost_graph_target(sym_graph_target, model):
        target_mean, target_std = model.get_normalization()
        sym_set_len = model.get_input_symbols()["set_lengths"]
        target_normalizing = 1.0 / target_std
        graph_target_normalized = (
            sym_graph_target - target_mean
        ) * target_normalizing

        graph_cost = tf.reduce_mean(
            (model.get_graph_out_normalized() - graph_target_normalized) ** 2,
            name="graph_cost",
        )

        return graph_cost

    def step(self, session, step, probe_count=1000):
        input_data = self.batchloader.get_train_batch(self.batch_size, probe_count=probe_count)
        feed_dict = {}
        for key in self.input_symbols.keys():
            feed_dict[self.input_symbols[key]] = input_data[key]
        feed_dict[self.sym_learning_rate] = self.get_learning_rate(step)
        session.run([self.train_op], feed_dict=feed_dict)
