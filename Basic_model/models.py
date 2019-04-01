# -*- coding: utf-8 -*-
from layers import *

# from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class mGLAD(Model):
    # inputs: placeholders['edges'] 形状为K
    # outputs: 形状为task节点数*总类数x

    def __init__(self, placeholders, worker_num, task_num, input_dim, edge_type, ability_num, **kwargs):
        super(mGLAD, self).__init__(**kwargs)
        self.edge_type = edge_type
        self.ability_num = ability_num
        self.inputs = placeholders['edges']
        self.input_dim = input_dim
        self.worker_num = worker_num
        self.task_num = task_num
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = [worker_num, task_num, edge_type]  # placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        def Cal_ProbLoss(loss, P, edges):
            # predict_edges' shape:(K*x)
            def cond_worker(i, loss_now):
                # 判断第i个worker
                return i < edges.shape[0]

            def body_worker(i, loss_now):
                # 对loss进行累加运算
                def cond_task(j, loss_now):
                    # 判断第j个task
                    return j < edges.shape[1]

                def body_task(j, loss_now):
                    # 对loss进行累加运算
                    if(edges[i][j]==-1):
                        return j+1,loss_now
                    loss_now = tf.add(loss_now, P[i][j][edges[i][j]])
                    # loss_now = tf.add(loss_now, tf.log(1- P[i][j][edges[i][j]]))
                    return j + 1, loss_now

                _, loss_now = tf.while_loop(cond_task, body_task, [0, loss_now])
                return i + 1, loss_now

            _, loss = tf.while_loop(cond_worker, body_worker, [0, loss])
            return loss

        # Weight decay loss
        with tf.name_scope('loss'):
            for var in self.layers[0].Vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

            # Cross entropy error

            # 此处计算输出边的连接关系和原本边的关系的交叉熵
            # 但是原文意思是要计算概率，也就是每条边和原本相同的值的概率
            # TO DO：确定outputs的输出形状，如果原始边连接矩阵的形状是K，总共有x个label，那就应该是K*x，然后softmax找原始Label概率

            self.loss += -1*tf.log(Cal_ProbLoss(self.loss, self.outputs, self.placeholders['edges']))
            # tf.scalar_summary('loss', self.loss)

            tf.summary.scalar('loss', self.loss)

    def _accuracy(self):
        print(tf.argmax(self.outputs, 2).dtype)
        print(self.placeholders['edges'].dtype)
        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(tf.cast(
                tf.equal(tf.cast(tf.argmax(self.outputs, 2), float), tf.cast(self.placeholders['edges'], float)),
                tf.float32), name='accuracy'
            )
            tf.summary.scalar('accuracy', self.accuracy)

    def _build(self):
        print("[ model ] Building mGLAD model......")
        print("[ model ] Appending MPNN layer......")
        self.layers.append(mpnn(input_dim=self.input_dim,
                                edge_type=self.edge_type,
                                ability_num=self.ability_num,
                                task_num=self.task_num,
                                worker_num=self.worker_num,
                                output_dim=[self.placeholders['worker_num'] + self.placeholders['task_num'],
                                            self.placeholders['ability_num'] + self.placeholders['edge_type']],
                                placeholders=self.placeholders,
                                update_step=5,
                                logging=self.logging))
        print("[ model ] Appending Decoder layer......")
        self.layers.append(Decoder(input_dim=[self.placeholders['worker_num'] + self.placeholders['task_num'],
                                              self.placeholders['ability_num'] + self.placeholders['edge_type']],
                                   edge_type=self.edge_type,
                                   worker_num=self.worker_num,
                                   task_num=self.task_num,
                                   ability_num=self.ability_num,
                                   output_dim=self.output_dim,
                                   placeholders=self.placeholders))
        print("[ model ] Build finished.")

    def predict(self):
        return tf.argmax(self.outputs, axis=2, name='predict')
