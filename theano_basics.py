"""
Baby steps in Theano library

"""
import numpy
from matplotlib import pyplot as plt
import theano.tensor as T
from theano import function
from theano import pp


'''
this is a brief insight of how the theano library works

everything we do, every operation, instead of a usual for example "x + y" is declared differently

variables are first declared as Theano based ones and then we assemble an expression as theano function

the reason for this is that theano creates a C code in the backend to make an operation

then the operation's code is optimized by compiler

following code shows a baseline on how to create simple equations, though it may seem intuitive, it
does things in quite unusual way 
'''

x = T.dscalar('x')
y = T.dscalar('y')
z = x + y
f = function([x, y], z)
print(f(2, 3))
print(numpy.allclose(f(16.3, 12.1), 28.4))

'''
here we defined simple addition function in more steps than usual

it's because theano acts on symbols to perform creation of C code that runs in background

we declare 2 "scalars" (numbers), assigned them to x and y, respectively, then we
hinted the operation (addition) that gets stored in z variable.

a function declaration takes inputs and outputs, in our case it takes an array [x, y]
and returns z. Z is evaluated as addition
'''

print(pp(z))
'''
here we see how theano assigned the operation
'''

print(z.eval({x : 16.3, y : 12.1}))
'''
we could also invoke the "eval()" from a z variable that had addition operation
"eval()" takes a dictionary with names of variables and values to be assigned to them

eval() utimately imports a "function()", so we end up in the same situation, so it is
slower the first time we invoke this, subsequent invocations are faster since it saves
"function()" imported already

that way we don't need to import "function()", but importing and using it is more 
flexible than relyin on "eval()" itself
'''

'''
Addition of two matrices is also very simple, the only difference is using 
T.dmatrix instead of T.dscalar
'''

x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f2 = function([x, y], z)

'''
now we do not assign the matrices to x and y directly, but instead we pass matrices 
to function as variables, invoking it like: "f2([matrix_nr_1], [matrix_nr_2])

here we add, elementwise, 2 2-dimensional matrices
'''

a1 = [[1, 2],
      [3, 4]]
a2 = [[10, 20],
      [30, 40]]

result = f2(a1, a2)
print(result)

'''
we could also utilize numpy.array
'''
a3 = numpy.array([[1, 2],
                  [3, 4]])
a4 = numpy.array([[10, 20],
                  [30, 40]])

print(f2(a3, a4))

'''
ex1. modify the code at the bottom, 

import theano
a = theano.tensor.vector() # declare variable
out = a + a ** 10
# build symbolic expression
f = theano.function([a], out)
# compile function
print(f([0, 1, 2]))

to reflect the expression: a ** 2 + b ** 2 + 2 * a * b
'''

c = 16
d = 13

sample_result = [c**2 + d**2 + 2*c*d]

a = T.vector('a') # declare variable
b = T.vector('b')
add = a + b
out = T.power(add, 2)
# build symbolic expression
f3 = function([a, b], out)
# compile function
print(f3([0, 1, 2], [0, 1, 2]))

assert sample_result == f3([16], [13])
print(f3([16], [13]))

'''
calculation more elaborate functions is also possible
here we calculate a result of a "logistic" function, sometimes referred to as "sigmoid"

'''

matrix = T.dmatrix('matrix')
s = 1/ (1+ T.exp(-matrix))
sigmoid = function([matrix], s)
X = numpy.linspace(-6, 6, 100)
values = sigmoid([X])[0]

print(X)
print(values)

plt.plot(X, values)
plt.show()


'''
We could also produce gradients.
Theano provides efficient symbolic differentiaition, using T.grad as a macro

lets compute a gradient of logistic function

note that we compute it for SCALAR values here!!!
'''

x2 = T.scalar('x2')
sigm = 1/ (1+ T.exp(-x2))
sigm_grad = T.grad(sigm, x2)
# uncomment this if you copy code to another program, i already declared this above
# X = numpy.linspace(-6, 6, 100)
'''
ok we declared derivative (gradient) and a regular function, at the coordinate passed
to the function

theano "function()" can also accept multiple outputs, not only multiple inputs

here we calculate the regular function output, as well as the derivative at the same point

KNOWN PROBLEM: the result is a list of 0-dimensional arrays, which can't be accessed in regular way
use "list[index].sum()" to extract a desired value from it
'''

multiple_sigmoid_outputs = function([x2], [sigm, sigm_grad])

val_out = multiple_sigmoid_outputs(4)
print(val_out)
print(val_out[0].sum(), val_out[1].sum())

'''
ok so we defined a derivative of a function at a single point, but what about computing
whole lists of points?

you have to wrap the result of a regular with the "T.sum" to make that happen
'''


x3 = T.matrix('x3')
sigm2 = T.sum(1/ (1+ T.exp(-x3)))
sigm2_grad = T.grad(sigm2, x3)
logistics = function([x3], [sigm2, sigm2_grad])
values2 = logistics([X])

print(values2)
Y = values2[1][0]
plt.plot(X, Y, c='red')
plt.plot(X, values, c='blue')
plt.show()

print(logistics.maker.fgraph.outputs)
pp(logistics.maker.fgraph.outputs[0])
