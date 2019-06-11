from sklearn.utils.class_weight import compute_class_weight


if __name__ == '__main__':
    res = compute_class_weight('balanced', [0, 1, 2], [0, 1, 1, 2, 2, 2])
    print(res)