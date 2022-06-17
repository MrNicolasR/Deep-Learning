# Import Packages
import numpy as np

# Comp Graph function
def compGraph():
    # Create array
    arr = np.array([1,0,1,0,1,1,1,0,1])

    vec = np.array([1,0,1])


    # Node 1
    def nodeOne(a,b, mode, pos , b_in):
        if mode == 0:
            return np.dot(a,b)
        elif pos == 1:
            return np.dot(b_in, np.transpose(b))
        else:
            return np.dot(np.transpose(a), b_in)

    # Node 2
    def nodeTwo(a, mode, b_in):
        if mode == 0:
            return 1/ (1 + np.exp((-a)))
        else:
            return np.multiply((np.exp(-a)/np.square(1 + np.exp(-a))),b_in)

    # Node 3
    def nodeThree(a, mode, b_in):
        if mode == 0:
            return np.sum(np.square(a))
        else:
            return a * 2 * b_in

    # Reshape
    in_one = np.array(arr).reshape([3,3])
    in_two = np.array(vec).reshape([3,1])

    # Node 1
    out_one = nodeOne(in_one, in_two, 0, 0, 0)
    print("The Result of Node 1:", "\n", out_one)

    # Node 2
    out_two = nodeTwo(out_one, 0, 0)
    print("The Result of Node 2:", "\n", out_two)

    # Node 3
    out_three = nodeThree(out_two, 0, 0)
    print("The Result of the Computational Graph is:", "\n", out_three)

    # Backward Propagation
    print("Backward Propagation\n")

    # Node 3
    b_three = nodeThree(out_two, 1, 1)
    print("The Result of the Node 3:", "\n", out_three)

    # Node 2
    b_two = nodeTwo(out_one, 1, b_three)
    print("The Result of Node 2:", "\n", out_two)

    # Partial Differentiation with respect to w
    b_outw = nodeOne(in_one, in_two, 1, 1, b_two)
    print("The back ward differentiation output at Node 1 at position 1 result partial diff of output W : ","\n", b_outw)

    # Partial Differentiation with respect to x
    b_out = nodeOne(in_one, in_two, 1, 2, b_two)
    print("The back ward differentiation output at Node 1 at position 2 result partial diff of output X: ","\n", b_out)


if __name__ == "__main__":
    compGraph()