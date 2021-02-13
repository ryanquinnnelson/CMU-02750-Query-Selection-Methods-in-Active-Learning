import iwal.iwal_functions


def test_append_history_dictionary_matches():
    h = {
        'X': [],
        'y': [],
        'c': [],
        'Q': []
    }

    x_t = [1, 2]
    y_t = 1
    p_t = 0.5
    q_t = 1

    iwal.iwal_functions.append_history(h, x_t, y_t, p_t, q_t)

    # assert x_t in h['X']
