import numpy as np
from w2v_utils import *

words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')


def cosine_similarity(u, v):
    if np.all(u == v):
        return 1

    dot = None

    norm_u = None

    norm_v = None

    if np.isclose(norm_u * norm_v, 0, atol=1e-32):
        return 0

    cosine_similarity = None

    return cosine_similarity
