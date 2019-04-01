# -*- coding: utf-8 -*-
# from gcn.inits import *
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1. / keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class mpnn(Layer):
    # input_dim=(39*104)
    # output_dim=[(39*D),(104*L)]

    def __init__(self, input_dim, task_num, worker_num, edge_type, ability_num, output_dim, placeholders, update_step,
                 **kwargs):
        super(mpnn, self).__init__(**kwargs)
        # self.update_function_a = update_a
        # self.update_function_t = update_t
        self.placeholders = placeholders
        self.adj = placeholders['edges']
        self.step = update_step
        self.input_dim = input_dim
        self.ability_num = ability_num
        self.edge_type = edge_type
        self.Vars = {}
        self.task_num = task_num
        self.worker_num = worker_num
        ## 定义基本变量
        ## adj: 邻接矩阵
        ## step: 更新次数
        ## output: 输出 

        with tf.variable_scope('mpnn_vars'):
            # 这个地方要加入可训练参数，在mpnn层里的

            # TO DO: 确定形状
            # 此处形状待定，暂时先按下面的来。

            self.Vars["Awij"] = tf.Variable(
                initial_value=tf.truncated_normal(shape=[edge_type, ability_num, edge_type], mean=0, stddev=0.01),
                name="Awij")
            # 在这里的形状是（ability类别*edge类别），即10*2

            self.Vars["Awij2"] = tf.Variable(
                initial_value=tf.truncated_normal(shape=[edge_type, edge_type, ability_num], mean=0, stddev=0.01),
                name="Awij2")
            # 在这里的形状是（1*每个人的ability类别），此处是1*10

        # if self.logging:
        #    self._log_vars()

    def _call(self, inputs):
        # 进行调用，会把输入图给导进来，然后调用下面的更新函数
        # TODO : 
        # 1.务必确定初始的a,t形状
        # 2.务必确定更新函数的写法，有些复杂。

        # 难点：
        # **更新a：是否要把不同label的累加矩阵算上
        # **更新t：确定只加正确类的label的权重

        first_a = tf.random_normal(shape=[self.worker_num, self.placeholders['ability_num']], stddev=1, seed=1,
                                   name='first_a')
        first_t = tf.random_normal(shape=[self.task_num, self.placeholders['edge_type']], stddev=1, seed=1,
                                   name='first_t')

        def cond(i, update_a, update_t):
            return i < self.step

        def body(i, update_a, update_t):
            def cond_a(j, update_a, update_t):
                # 判断a的更新进行完毕与否
                return j < self.worker_num

            def cond_t(k, update_a, update_t):
                # 判断t的更新进行完毕与否

                return k < self.task_num

            def body_a(j, update_a, update_t):
                # 用于循环迭代每一轮a的更新
                def cond_tau(jj,count,  update_a, update_t):
                    # 用于判断是否迭代完了每一个worker

                    return jj < self.task_num

                def body_tau(jj, count, update_a, update_t):
                    # 用于针对每一个worker的ability，都根据原始label选择对应矩阵相乘
                    if(inputs[j][jj]==-1):
                        return jj + 1, count, update_a, update_t
                    now_t = update_t[jj]  # 根据这条边是啥取用指定tau里的值，维度 1*2

                    A2_label = self.Vars["Awij2"][inputs[j][jj]]
                    # 根据不同label 选用不同的A2

                    update_a = tf.reshape(
                        tf.concat(
                            [
                                update_a[:j],
                                tf.add(update_a[j],
                                       tf.matmul(tf.reshape(now_t, [1, self.edge_type]), A2_label),
                                       name='each_tau_x_its_A2'),
                                update_a[j + 1:]
                            ], axis=0, name='now_update_a'),
                        (self.worker_num, -1)
                    )
                    # update_a_final=tf.concat(update_a_final,update_aj,axis=0)
                    # 将update_a的第j个worker的得分进行一个累加计算

                    return jj + 1, count+1, update_a, update_t

                _,count,update_a, update_t = tf.while_loop(cond_tau, body_tau, [0, 0,update_a, update_t],
                                                      name='MPNN_each_t_for_a')
                update_a=update_a*(tf.cast(self.task_num,tf.float32)/tf.cast(count,tf.float32))
                #不知道为啥，一除就凉，变瞎猜。

                return j + 1, update_a, update_t

            def body_t(k, update_a, update_t):
                # 用于循环迭代每一轮t的更新,迭代task个数次，是第二个维度

                def cond_aj(kk, update_a, update_t):
                    # 用于判断是否考虑到了每一个worker
                    return kk < self.worker_num

                def body_aj(kk, update_a, update_t):
                    if(inputs[kk][k]==-1):
                        return kk + 1, update_a, update_t
                    A_label = self.Vars["Awij"][inputs[kk][k]]
                    # 根据不同label取用不同矩阵，存在本次A_label中

                    update_t = tf.reshape(
                        tf.concat(
                            [
                                update_t[:k],
                                tf.add(update_t[k],  # 求和的过程，从kk=0到kk=最后一个worker的序号，都给加上
                                       tf.matmul(tf.reshape(update_a[kk], shape=[-1, self.ability_num]),
                                                 A_label), name='each_a_x_its_A'),
                                update_t[k + 1:]
                            ],
                            axis=0), (self.task_num, -1), name='now_update_t')

                    # 1*10 x 10*2  = 1*2
                    return kk + 1, update_a, update_t

                _, update_a, update_t = tf.while_loop(cond_aj, body_aj, [0, update_a, update_t],
                                                      name='MPNN_each_a_for_t')
                # tf.Print(update_a,["a:",update_a])

                return k + 1, update_a, update_t

            # TO DO:这个地方维度计算没有敲定，还有转置等操作要做
            _, update_a, update_t = tf.while_loop(cond_a, body_a, [0, update_a, update_t], name='MPNN_a_update')
            # 对a进行一轮更新

            _, update_a, update_t = tf.while_loop(cond_t, body_t, [0, update_a, update_t], name='MPNN_t_update')
            # 对t进行一轮更新
            # update_t = tf.nn.softmax(update_t)

            return i + 1, update_a, tf.nn.softmax(update_t)

        i, final_a, final_t = tf.while_loop(cond, body, [0, first_a, first_t], name='MPNN_main')
        self.a = final_a
        self.t = final_t  # tf.nn.softmax(final_t)

        padding_t = tf.random_normal(shape=[self.task_num, self.ability_num])
        padding_a = tf.random_normal(shape=[self.worker_num, self.edge_type])
        # 此处为了保证输出是一个完整的张量,于是分别对a,t进行padding操作，让它们的维度均为：[-1, worker特征数 + task特征数]
        self.output = tf.concat([tf.concat([final_a, padding_a], axis=1), tf.concat([final_t, padding_t], axis=1)],
                                axis=0)
        # 拼接后即可传出output
        # 维度为：（ worker node+ task node , worker feature, task feature ）

        return self.output


class Decoder(Layer):
    def __init__(self, input_dim, task_num, worker_num, output_dim, ability_num, edge_type, placeholders, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.placeholders = placeholders
        self.ability_num = ability_num
        self.edge_type = edge_type
        self.worker_num = worker_num
        self.task_num = task_num
        self.P = tf.random_normal(shape=[self.worker_num, self.task_num, edge_type], stddev=1, seed=1)
        self.Vars = {}

        with tf.variable_scope('Decoder_vars'):
            self.Vars["W"] = tf.Variable(
                initial_value=tf.truncated_normal(shape=[self.ability_num, 1], mean=0, stddev=0.01), name="W")
            # 在这里的形状是（ability类别*1），即10*1
            self.Vars["b"] = tf.Variable(initial_value=tf.truncated_normal(shape=[1], mean=0, stddev=0.01), name="b")
            # 在这里的形状是 1

    def _call(self, inputs):
        # print(self.ability_num)
        # print(self.input_dim)
        worker_feature = inputs[:self.worker_num, :self.ability_num]
        task_feature = inputs[self.worker_num:, :self.edge_type]
        # print(worker_feature.shape)
        print("inputs' shape:", inputs.shape)
        print("workers' features' shape:", worker_feature.shape)
        print("tasks' features' shape:", task_feature.shape)

        # 进入循环，此处循环较为复杂，要进行每个worker 对每个task 的每个label的可能性计算

        def cond_worker(i, P):
            # 针对第i个worker

            return i < self.worker_num

        def body_worker(i, P):
            def cond_task(j, P):
                # 针对第j个task

                return j < self.task_num

            def body_task(j, P):
                def cond_label(l, P):
                    # 针对第l个label
                    return l < self.edge_type

                def body_label(l, P):
                    # 对每个l进行计算
                    tau_prob = task_feature[j, l]
                    prob_part1 = tf.sigmoid(
                        tf.add(
                            tf.matmul(
                                tf.reshape(
                                    worker_feature[i, :self.ability_num], shape=[1, self.ability_num]),
                                self.Vars["W"]),
                            self.Vars["b"]),
                        name='prob_part_1'
                    )
                    prob_part2 = tf.divide(
                        tf.subtract(tf.constant(1.), prob_part1),
                        tf.cast(self.placeholders["edge_type"] - 1, tf.float32), name='prob_part_2')
                    P_l_single = tf.reshape(tf.matmul(
                        tf.pow(prob_part1, tau_prob),
                        tf.pow(
                            prob_part2,
                            tf.subtract(
                                tf.cast(tf.constant(1), tf.float32),
                                tau_prob)
                        )
                    ), shape=[1], name='P_1_single')

                    # 先计算出当前位置的P，是第i个worker给第j个label打第l个label的可能性
                    # 下面进行一个个赋值
                    # 首先进行对第三维度进行赋值，即label维度，合并，成为一个一维的向量

                    P_l = tf.reshape(
                        tf.concat([
                            P[i, j, :l],
                            P_l_single,
                            P[i, j, l + 1:]
                        ],
                            axis=0),
                        shape=[-1, self.edge_type]
                        , name='P_for_each_l')

                    # 接着对task维度进行赋值，合并，成为一个二维的向量，
                    # 代表这个worker对所有task的所有label打分的可能性
                    print(P_l.shape)

                    P_j = tf.reshape(
                        tf.concat([
                            P[i, :j],
                            P_l,
                            P[i, j + 1:]
                        ],
                            axis=0),
                        shape=[-1, self.task_num, self.edge_type]

                        , name='P_for_each_worker')

                    # 及接着对worker维度进行赋值，合并，成为一个三维向量，
                    # 代表所有worker对所有task的所有label打分的可能性
                    print(P_j.shape)

                    P = tf.reshape(
                        tf.concat([
                            P[:i],
                            P_j,
                            P[i + 1:]
                        ],
                            axis=0),
                        shape=[self.worker_num, self.task_num, self.edge_type]
                        , name='P_for_all')
                    return l + 1, P

                _, P = tf.while_loop(cond_label, body_label, [0, P], name='P_for_each_label')
                print('[info ] P\'s shape is:', P.shape)

                return j + 1, P

            _, P = tf.while_loop(cond_task, body_task, [0, P], name='P_for_each_task')

            return i + 1, P

        _, P = tf.while_loop(cond_worker, body_worker, [0, self.P], name='P_for_all_workers')
        # print('[info ] P\'s shape is:', P.shape)
        return P
        # 进行调用
