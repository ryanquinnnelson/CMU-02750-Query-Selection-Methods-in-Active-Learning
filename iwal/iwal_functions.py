
def append_history(history, x_t, y_t, p_t, q_t):
    """
    Adds new sample to given query history.

    :param history:
    :param x_t:
    :param y_t:
    :param p_t:
    :param q_t:
    :return:
    """
    # validate that history contains required keys
    if 'X' in history and 'y' in history and 'c' in history and 'Q' in history:
        history['X'].append(x_t)
        history['y'].append(y_t)
        history['c'].append(p_t)
        history['Q'].append(q_t)
    else:
        raise ValueError('history dictionary does not contain the required keys: X,y,c,Q')
