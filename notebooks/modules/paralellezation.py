import time
import multiprocessing

def f(x):
    return x**2

def g(x):
    time.sleep(1) # simulate heavy computations
    return x**2

if __name__ == '__main__':

    # a list of 'problems' to solve in parallel
    problems = range(8)

    # starting a multiprocessing pool and measure time
    pool = multiprocessing.Pool()
    time_start = time.time()
    result = pool.map(g, problems)
    time_stop = time.time()

    # print result
    execution_time = time_stop - time_start
    print 'executed %d problems in %d seconds' % (len(problems), execution_time)