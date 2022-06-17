# Import packages
import math

# Computational Graph Input
x1 = 1
x2 = 2
w1 = 3
w2 = 4

float(x1)
float(x2)
float(w1)
float(w2)


# Comp Graph Function
def compGraph():
    # Node 1
    def nodeOne(a, b, mode, pos, b_input):
        if mode == 0:
            return a * b
        elif pos == 1:
            return b * b_input
        else:
            return a * b_input


    # Node 2
    def nodeTwo(a, mode, b_input):
        if mode == 0:
            return math.cos(a)
        else:
            return -(math.sin(a)) * b_input


    # Node 3
    def nodeThree(a, b, mode, pos, b_input):
        if mode == 0:
            return a * b
        elif pos == 1:
            return b * b_input
        else:
            return a * b_input


    # Node 4
    def nodeFour(a, mode, b_input):
        if mode == 0:
            return math.sin(a)
        else:
            return math.cos(a) * b_input


    # Node 5
    def nodeFive(a, mode, b_input):
        if mode == 0:
            return a * a
        else:
            return 2 * a * b_input


    # Node 6
    def nodeSix(a, b, mode, b_input):
        if mode == 0:
            return a + b
        else:
            return 1 * b_input


    # Node 7
    def nodeSeven(a, b, mode, b_input):
        if mode == 0:
            return a + b
        else:
            return 1 * b_input


    # Node 8
    def nodeEight(a, mode, b_input):
        if mode == 0:
            return 1 / a
        else:
            return -1 / (a * a)
    # Node 1
    outOne = nodeOne(x2,w2,0,0,0)
    print("The result of Node 1: %f\n" % outOne)

    # Node 2
    outTwo = nodeTwo(outOne,0,0)
    print("The result of Node 2: %f\n" % outTwo)

    # Node 3
    outThree = nodeThree(x1, w1,0,0,0)
    print("The result of Node 3: %f\n" % outThree)

    # Node 4
    outFour = nodeFour(outThree,0,0)
    print("The result of Node 4: %f\n" % outFour)

    # Node 5
    outFive = nodeFive(outFour,0,0)
    print("The result of Node 6: %f\n" % outFive)

    # Node 6
    outSix = nodeSix(outFive,outTwo,0,0)
    print("The result of Node 6: %f\n" % outSix)

    # Node 7
    outSeven = nodeSeven(2,outSix,0,0)
    print("The result of Node 7: %f\n" % outSeven)

    # Node 8
    outEight = nodeEight(outSeven,0,0)
    print("The result of the Graph: %f\n" % outEight)

    print("\nBackward Propagation\n")

    # Node 8
    b_eight = nodeEight(outSeven,1,0)
    print("The result of Node 8: %f\n" % b_eight)

    # Node 7
    b_seven = nodeSeven(2,outSix,1,b_eight)
    print("The result of Node 7: %f\n" % b_seven)

    # Node 6
    b_six = nodeSix(outFive, outTwo, 1, b_seven)
    print("The result of Node 6: %f\n" % b_six)

    # Node 5
    b_five = nodeFive(outFour, 1, b_six)
    print("The result of Node 5: %f\n" % b_five)

    # Node 4
    b_four = nodeFour(outThree, 1, b_five)
    print("The result of Node 4: %f\n" % b_four)

    # Node 3
    b_three_x1 = nodeThree(x1, w1, 1, 1, b_four)
    b_three_w1 = nodeThree(x1,w1, 0, 2, b_four)
    print("Backward differential output at Node 3 position 1 output with respect to x1:  %f\n" % b_three_x1)
    print("Backward differential output at Node 3 position 2 output with respect to w1:  %f\n" % b_three_w1)

    # Node 2
    b_two = nodeTwo(outOne, 1, b_six)
    print("Backward differential output at Node 2:  %f\n" % b_two)

    # Node 1
    b_one_x2 = nodeOne(x2, w2, 1, 1, b_two)
    b_one_w2 = nodeOne(x2, w2, 1, 2, b_two)
    print("Backward differential output at Node 1 with respect to x2: %f\n" % b_one_x2)
    print("Backward differential output at Node 1 with respect to w2: %f\n" % b_one_w2)

    # Final results
    print("The Result of the partial differentiation with respect to x1: %f\n" % b_three_x1)

    print("The Result of the partial differentiation with respect to w1: %f\n" % b_three_w1)

    print("The Result of the partial differentiation with respect to x2: %f\n" % b_one_x2)

    print("The Result of the partial differentiation with respect to w2: %f\n" % b_one_w2)

if __name__ == "__main__":
    compGraph()