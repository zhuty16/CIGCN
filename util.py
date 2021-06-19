import numpy as np
import scipy.sparse as sp


def get_adj_matrix(train_dict, validate_dict, rel_dict, num_user, num_item, test=0):
    row = [u for u in train_dict for i in train_dict[u]] + [i+num_user for u in train_dict for i in train_dict[u]] + [i+num_user for i in rel_dict for j in rel_dict[i]] + [j+num_user for i in rel_dict for j in rel_dict[i]]
    col = [i+num_user for u in train_dict for i in train_dict[u]] + [u for u in train_dict for i in train_dict[u]] + [j+num_user for i in rel_dict for j in rel_dict[i]] + [i+num_user for i in rel_dict for j in rel_dict[i]]
    if test == 1:
        row += [u for u in validate_dict] + [validate_dict[u]+num_user for u in validate_dict]
        col += [validate_dict[u]+num_user for u in validate_dict] + [u for u in validate_dict]
    rel_matrix = sp.coo_matrix(([1] * len(row), (row, col)), (num_user+num_item, num_user+num_item)).astype(np.float32)
    row_sum = np.array(rel_matrix.sum(1)) + 1e-24
    degree_mat_inv_sqrt = sp.diags(np.power(row_sum, -0.5).flatten())
    rel_matrix_normalized = degree_mat_inv_sqrt.dot(rel_matrix.dot(degree_mat_inv_sqrt)).tocoo()
    indices = np.vstack((rel_matrix_normalized.row, rel_matrix_normalized.col)).transpose()
    values = rel_matrix_normalized.data.astype(np.float32)
    shape = rel_matrix_normalized.shape
    return indices, values, shape


def get_train_data(train_dict, num_item):
    train_data = list()
    for u in train_dict:
        for i in train_dict[u]:
            neg_sample = np.random.randint(num_item)
            while neg_sample in train_dict[u]:
                neg_sample = np.random.randint(num_item)
            train_data.append([u, i, neg_sample])
    return train_data


def get_train_batch(train_data, batch_size):
    train_batch = list()
    np.random.shuffle(train_data)
    i = 0
    while i < len(train_data):
        train_batch.append(np.asarray(train_data[i:i+batch_size]))
        i += batch_size
    return train_batch


def get_test_data(test_dict, negative_dict):
    test_data = [[u, test_dict[u]] + negative_dict[u] for u in test_dict]
    return np.asarray(test_data)


def get_feed_dict(model, batch_data, adj_matrix, emb_dropout_rate, node_dropout_rate):
    feed_dict = dict()
    feed_dict[model.u] = batch_data[:, 0]
    feed_dict[model.i] = batch_data[:, 1]
    feed_dict[model.j] = batch_data[:, 2]
    feed_dict[model.adj_matrix] = adj_matrix
    feed_dict[model.emb_dropout_rate] = emb_dropout_rate
    feed_dict[model.node_dropout_rate] = node_dropout_rate
    return feed_dict


def get_feed_dict_test(model, batch_data_test, adj_matrix_test):
    feed_dict = dict()
    feed_dict[model.u] = batch_data_test[:, 0]
    feed_dict[model.test_item] = batch_data_test[:, 1:]
    feed_dict[model.adj_matrix] = adj_matrix_test
    feed_dict[model.emb_dropout_rate] = 0.0
    feed_dict[model.node_dropout_rate] = 0.0
    return feed_dict
