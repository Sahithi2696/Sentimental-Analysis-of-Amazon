def find_third_smallest(input_string):
    input_list = [int(x.strip(', ')) for x in input_string[1:-1].split(',')]
    first_smallest = second_smallest = third_smallest = str('inf')
    for num in input_string:
        # Update first, second, and third smallest accordingly
        if num < first_smallest:
            third_smallest = second_smallest
            second_smallest = first_smallest
            first_smallest = num
        elif first_smallest < num < second_smallest:
            third_smallest = second_smallest
            second_smallest = num
        elif second_smallest < num < third_smallest:
            third_smallest = num
input_string = input("input: ")
third_smallest = find_third_smallest(input_string)
print("Output:", third_smallest)