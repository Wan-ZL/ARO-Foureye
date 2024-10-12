from multiprocessing import cpu_count
from multiprocessing import Process
from multiprocessing import Manager


def loop(cpu, string, result):
    for i in range(10000):
        print(f"cpu {cpu}: {i}\n" + string)
    result[cpu] = cpu


if __name__ == '__main__':
    threads = []
    result = Manager().dict()
    for i in range(cpu_count()):
        t = Process(target=loop, args=[i, "hahaha", result])
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    print(result)

