'''
GCN4Rec
@author: Tianyu Zhu
@created: 2021/6/16
'''
import tensorflow as tf


class CIGCN(object):
    def __init__(self, num_user, num_item, args):
        self.num_user = num_user
        self.num_item = num_item

        self.num_factor = args.num_factor
        self.num_layer = args.num_layer
        self.l2_reg = args.l2_reg
        self.lr = args.lr

        self.adj_matrix = tf.sparse_placeholder(tf.float32, name="adj_matrix")
        self.u = tf.placeholder(tf.int32, [None], name="uid")
        self.i = tf.placeholder(tf.int32, [None], name="iid")
        self.j = tf.placeholder(tf.int32, [None], name="jid")
        self.node_dropout_rate = tf.placeholder(tf.float32, name="node_dropout_rate")
        self.emb_dropout_rate = tf.placeholder(tf.float32, name="emb_dropout_rate")

        adj_matrix_dropout = self.node_dropout(self.adj_matrix, tf.shape(self.adj_matrix.values)[0], 1-self.node_dropout_rate)

        with tf.name_scope('embedding_table'):
            embedding = tf.Variable(tf.random_normal([self.num_user + self.num_item, self.num_factor], stddev=0.01, name="embedding"))

        with tf.name_scope('graph_convolution'):
            embedding_final = [embedding]
            layer = embedding
            
            if args.model == 'CIGCN':
                W = []
                for k in range(self.num_layer):
                    W.append(tf.Variable(tf.random_normal([1, self.num_factor], stddev=0.01), name="W"+str(k)))
                    layer = tf.nn.tanh(tf.sparse_tensor_dense_matmul(adj_matrix_dropout, layer) * W[k])
                    layer = tf.nn.dropout(layer, keep_prob=1-self.emb_dropout_rate)
                    embedding_final += [layer]
                embedding_final = tf.stack(embedding_final, 1)
                embedding_final = tf.reduce_sum(embedding_final, 1)
                
            elif args.model == 'LightGCN':
                for k in range(self.num_layer):
                    layer = tf.sparse_tensor_dense_matmul(adj_matrix_dropout, layer)
                    embedding_final += [layer]
                embedding_final = tf.stack(embedding_final, 1)
                embedding_final = tf.reduce_mean(embedding_final, 1)
                
            elif args.model == 'LR-GCCF':
                for k in range(self.num_layer):
                    layer = tf.sparse_tensor_dense_matmul(adj_matrix_dropout, layer)
                    embedding_final += [layer]
                embedding_final = tf.concat(embedding_final, 1)
                
            elif args.model == 'NGCF':
                W_1 = []
                W_2 = []
                for k in range(self.num_layer):
                    W_1.append(tf.Variable(tf.random_normal([self.num_factor, self.num_factor], stddev=0.01), name="W_1"+str(k)))
                    W_2.append(tf.Variable(tf.random_normal([self.num_factor, self.num_factor], stddev=0.01), name="W_2"+str(k)))
                    layer = tf.nn.leaky_relu(tf.matmul(tf.sparse_tensor_dense_matmul(adj_matrix_dropout, layer), W_1[k])
                                             + tf.matmul(layer, W_1[k])
                                             + tf.matmul(tf.sparse_tensor_dense_matmul(adj_matrix_dropout, layer * layer), W_2[k]))
                    layer = tf.nn.dropout(layer, keep_prob=1-self.emb_dropout_rate)
                    embedding_final += [layer]
                embedding_final = tf.concat(embedding_final, 1)
                
            else:
                embedding_final = embedding

            user_embedding, item_embedding = tf.split(embedding_final, [self.num_user, self.num_item], 0)

        with tf.name_scope('prediction'):
            u_emb = tf.nn.embedding_lookup(user_embedding, self.u)
            i_emb = tf.nn.embedding_lookup(item_embedding, self.i)
            j_emb = tf.nn.embedding_lookup(item_embedding, self.j)
            r_hat_ui = tf.reduce_sum(u_emb * i_emb, 1, True)
            r_hat_uj = tf.reduce_sum(u_emb * j_emb, 1, True)

        with tf.name_scope('loss'):
            bpr_loss = -tf.reduce_mean(tf.log_sigmoid(r_hat_ui - r_hat_uj))
            self.loss = bpr_loss + self.l2_reg * tf.reduce_sum([tf.nn.l2_loss(va) for va in tf.trainable_variables()])
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        with tf.name_scope('test'):
            self.test_item = tf.placeholder(tf.int32, [None, 1000])
            test_item_emb = tf.nn.embedding_lookup(item_embedding, self.test_item)
            test_logits = tf.matmul(tf.expand_dims(u_emb, 1), tf.transpose(test_item_emb, [0, 2, 1]))
            self.test_logits = tf.squeeze(test_logits, 1)

    def node_dropout(self, adj_matrix, num_value, keep_prob):
        noise_shape = [num_value]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(adj_matrix, dropout_mask) * tf.div(1.0, keep_prob)
        return pre_out
