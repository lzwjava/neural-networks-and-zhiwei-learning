import numpy as np
from w2v_utils import *

words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')


def cosine_similarity(u, v):
    if np.all(u == v):
        return 1

    dot = np.dot(u, v)

    norm_u = np.linalg.norm(u)

    norm_v = np.linalg.norm(v)

    if np.isclose(norm_u * norm_v, 0, atol=1e-32):
        return 0

    cosine_similarity = dot / norm_u / norm_v

    return cosine_similarity


father = word_to_vec_map["father"]
mother = word_to_vec_map["mother"]
ball = word_to_vec_map["ball"]
crocodile = word_to_vec_map["crocodile"]
france = word_to_vec_map["france"]
italy = word_to_vec_map["italy"]
paris = word_to_vec_map["paris"]
rome = word_to_vec_map["rome"]

print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
print("cosine_similarity(ball, crocodile) = ", cosine_similarity(ball, crocodile))
print("cosine_similarity(france - paris, rome - italy) = ", cosine_similarity(france - paris, rome - italy))


def cosine_similarity_test(target):
    a = np.random.uniform(-10, 10, 10)
    b = np.random.uniform(-10, 10, 10)
    c = np.random.uniform(-1, 1, 23)

    assert np.isclose(cosine_similarity(a, a), 1), "cosine_similarity(a, a) must be 1"
    assert np.isclose(cosine_similarity((c >= 0) * 1, (c < 0) * 1), 0), "cosine_similarity(a, not(a)) must be 0"
    assert np.isclose(cosine_similarity(a, -a), -1), "cosine_similarity(a, -a) must be -1"
    assert np.isclose(cosine_similarity(a, b), cosine_similarity(a * 2,
                                                                 b * 4)), "cosine_similarity must be scale-independent. You must divide by the product of the norms of each input"

    print("\033[92mAll test passed!")


cosine_similarity_test(cosine_similarity)


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    e_a, e_b, e_c = None

    words = word_to_vec_map.keys()
    max_cosine_sim = -100
    best_word = None

    for w in words:

        if w == word_c:
            continue

        cosine_sim = None

        if None > None:
            max_cosine_sim = None
            best_word = None

    return best_word


def complete_analogy_test(target):
    a = [3, 3]
    a_nw = [2, 4]
    a_s = [3, 2]

    c = [-2, 1]

    word_to_vec_map = {'a': a,
                       'synonym_of_a': a,
                       'a_nw': a_nw,
                       'a_s': a_s,
                       'c': c,
                       'c_n': [-2, 2],
                       'c_ne': [-1, 2],
                       'c_e': [-1, 1],
                       'c_se': [-1, 0],
                       'c_s': [-2, 0],
                       'c_sw': [-3, 0],
                       'c_w': [-3, 1],
                       'c_nw': [-3, 2]
                       }

    for key in word_to_vec_map.keys():
        word_to_vec_map[key] = np.array(word_to_vec_map[key])

    assert (target('a', 'a_nw', 'c', word_to_vec_map) == 'c_nw')
    assert (target('a', 'a_s', 'c', word_to_vec_map) == 'c_s')
    assert (target('a', 'synonym_of_a', 'c', word_to_vec_map) != 'c'), "Best word cannot be input query"
    assert (target('a', 'c', 'a', word_to_vec_map) == 'c')

    print("\033[92mAll tests passed")


complete_analogy_test(complete_analogy)
