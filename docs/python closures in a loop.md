In Python, closures can behave unexpectedly when capturing loop variables. This is due to Python's late binding behavior, where the value of the variable is looked up when the closure is called, not when it's defined.
Here's an example demonstrating the issue:
Python



Execution output
def create_funcs():
    funcs = []
    for i in range(3):
        def func():
            return i
        funcs.append(func)
    return funcs

my_funcs = create_funcs()
for func in my_funcs:
    print(func())
2
2
2
You might expect this to print 0 1 2, but it actually prints 2 2 2. This is because all three closures are referencing the same i variable, which is 2 at the end of the loop.
Solution:
To capture the value of the loop variable at each iteration, you can use a default argument:
Python



Execution output
def create_funcs():
    funcs = []
    for i in range(3):
        def func(i=i):  # Capture i as a default argument
            return i
        funcs.append(func)
    return funcs

my_funcs = create_funcs()
for func in my_funcs:
    print(func())
0
1
2
This will correctly print 0 1 2. The default argument creates a new variable within each closure's scope, capturing the value of i at the time the closure is created.



## about lambda

Using a lambda function in Python for the purpose of capturing variables in a closure, especially in a loop, can be tricky because lambda functions capture variables from their surrounding scope at the time they are executed, not when they are defined.


