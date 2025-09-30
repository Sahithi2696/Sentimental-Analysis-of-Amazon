# creating a function
def findIntersection(A,B):
    # declare empty set C
    C = set({})
    # iterate A
    for ele in A:
        # find common
        if ele in B:
            # store common ele in C
            C.add(ele)
    return C
# Driver code
if __name__ == '__main__':
    # creating two sets
    A = input("A:")
    str_set = set(A)
    print(str_set)
    B = input("B:")
    str_set1 = set(B)
    print(str_set1)
    # call the function and print returned value
    print(findIntersection(A,B))