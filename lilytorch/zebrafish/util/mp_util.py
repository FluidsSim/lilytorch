
from multiprocessing import Process, Semaphore, Queue
import numpy as np

def run_process(semaphore, result_queue, index, func, input_value):
    result = func(input_value)
    # Release semaphore before queue put to allow main process to receive from
    #queue while we're writing.
    semaphore.release()
    result_queue.put((index, result))


class UniqueProcessMap(object):
    """A multi-processing version of map which uses a new process for each job,
    rather than worker processes that are re-used."""

    def __init__(self, max_processes):
        self.semaphore = Semaphore(0)
        self.result_queue = Queue()
        self.max_processes = max_processes

    def map(self, func, inputs):
        inputs = list(inputs)
        num_unfinished = len(inputs)
        results = [None] * len(inputs)

        for i in range(min(self.max_processes, len(inputs))):
            input_index = len(inputs) - 1
            process = Process(target=run_process, 
                              args=(self.semaphore, 
                                    self.result_queue, 
                                    input_index, func, 
                                    inputs.pop()))
            process.start()

        while num_unfinished > 0:
            self.semaphore.acquire()
            result_index, result = self.result_queue.get()
            results[result_index] = result
            num_unfinished -= 1
            if len(inputs) > 0:
                input_index = len(inputs) - 1
                process = Process(target=run_process, 
                                  args=(self.semaphore, 
                                        self.result_queue, 
                                        input_index, func, 
                                        inputs.pop()))
                process.start()

        return results


def sweep_1d(fun, inputs, num_process=2):
    if len(inputs)>1 and num_process>1:
        processes = UniqueProcessMap(num_process)
        out = processes.map(fun, inputs)
        return out
    else:
        out = []
        for inp in inputs:
            out.append(fun(inp))
        return out


def sweep_2d(fun, pars1, pars2, num_process=2):
    inputs = []
    x, y = np.meshgrid(pars1, pars2)
    for count, xvalue in enumerate(x):
        for (i,j) in zip(xvalue, y[count]):
            inputs.append([i,j])
    return sweep_1d(fun, inputs, num_process=num_process)

