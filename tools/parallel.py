#TODO: create parallel tool for speed up the processing
from multiprocessing import Process, Queue, JoinableQueue
import time
from valuations.valuations import valuations

class parallel:
    """
    this function is a tool to parallel the computation
    @param target_func is which will speed up
    """
    def __init__(self, target_func):
        self.target = target_func

    def processor(self, data_queue, results_queue):
        while not data_queue.empty():
            data = data_queue.get(timeout=1)
            process = self.target.valuate(data[0],data[1])
            results_queue.put(process)
            data_queue.task_done()

    def multi_compute(self, num_process: int, data_to_process: list):
        start = time.time()
        data_queue = JoinableQueue()
        results_queue = Queue()
        final_results = []
        Processes = []

        pass

