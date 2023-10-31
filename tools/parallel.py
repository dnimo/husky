from multiprocessing import Process, Queue, JoinableQueue
import time


class Parallel:
    """
    this function is a tool to parallel the computation
    @param target_func, which will speed up
    using multi_compute function to parallel the computed process.
    The input is num_process & data_to_process, which means cpu cores and data.
    The input structure like: num_process = 1, data_to_process = [("text1", "text2"), ("text1", "text2")...]
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
        processes = []

        for i in data_to_process:
            data_queue.put(i)

        for _ in range(num_process):
            woker = Process(target=self.processor, args=(data_queue, results_queue))
            processes.append(woker)
            woker.start()

        data_queue.join()

        # waiting for complete
        for process in processes:
            process.join()

        while not results_queue.empty():
            result = results_queue.get(timeout=1)
            final_results.append(result)

        end = time.time()

        cost = end - start

        print(f"All data processed time cost is {cost = :.4f}")

if __name__ == '__main__':
    print("testing")