from numpy import array, dot


def inner_product(a, b) -> array:
    """Compute the inner product of two matrices.
    a: matrix of shape (m, n)
    b: matrix of shape (n, p)
    returns: matrix of shape (m, p)"""
    assert a.shape[1] == b.shape[0], "Dimension mismatch: a.shape[1] != b.shape[0]"
    output = array([[0 for _ in range(b.shape[1])] for _ in range(a.shape[0])])
    # for rows in a
    for i in range(0, a.shape[0]):
        # for columns in b
        for j in range(0, b.shape[1]):
            # for columns in a
            for k in range(0, a.shape[1]):
                # remember that columns in a == rows in b
                # hence why k is used to iterate through the rows in b
                #                           V
                output[i][j] += a[i][k] * b[k][j]
    print(f"Output: {output}")
    return output


def test_inner_product() -> None:
    """Test the inner_product function"""
    a = array([[1, 2], [3, 4], [5, 6]])
    b = array([[1, 2, 3], [4, 5, 6]])
    print(f"Inner product: {dot(a, b)}")
    assert (inner_product(a, b) == array(
        [[9, 12, 15], [19, 26, 33], [29, 40, 51]])).all()
    assert (dot(a, b).all() == array([
        [9, 12, 15],
        [19, 26, 33], [29, 40, 51]]).all())
    # test a huge matrix
    a = array([[i for i in range(512)] for _ in range(512)])
    b = array([[i for i in range(512)] for _ in range(512)])
    assert (inner_product(a, b) == dot(a, b)).all()
    print("All tests pass, congrats! 🎉")


def main() -> None:
    test_inner_product()


if __name__ == '__main__':
    main()
