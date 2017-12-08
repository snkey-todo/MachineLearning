import timeit

# sample
'''
分别使用三种方式计算1～1000的所有数平方和，观察花费的时间
'''
normal_py_sec = timeit.timeit('sum(x*x for x in xrange(1000))', number=10000)
naive_np_sec = timeit.timeit('sum(na*na)', setup="import numpy; na = numpy.arange(1000)", number=10000)
good_np_sec = timeit.timeit('na.dot(na)', setup="import numpy; na = numpy.arange(1000)", number=10000)

print("normal python: %f sec" %normal_py_sec)
print("naive numpy: %f sec" %naive_np_sec)
print("good numpy: %f sec" %good_np_sec)


