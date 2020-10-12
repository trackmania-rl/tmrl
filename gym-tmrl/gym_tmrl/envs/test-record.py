import numpy as np
import cv2
import os
import mss
import time
import multiprocessing
from multiprocessing import Pool

sct = mss.mss()

# def grab_image(idx, queue, monitor):
#     print("BLA")
#     for _ in range(10):
#         queue.put((idx, np.asarray(sct.grab(monitor))[:, :, :3]))
#     print("BLout")


def grab_image(monitor):
    # print("BLA")
    return np.asarray(sct.grab(monitor))[:, :, :3]


if __name__ == "__main__":
    monitor = {"top": 30, "left": 0, "width": 500, "height": 250}

    jobs = []
    #queue = multiprocessing.Queue()
    pool = Pool(processes=4)
    t1 = time.time()
    for _ in range(100):
        data = pool.map(grab_image, [monitor, monitor, monitor, monitor])
    t2 = time.time()
    pool.close()

    # for i in range(4):
    #     p = multiprocessing.Process(target=grab_image, args=(i, queue, monitor,))
    #     jobs.append(p)
    #     p.start()
    # print("DEBUG: waiting for processes to terminate")
    # for p in jobs:
    #     p.join()


    print(f"time:{(t2-t1)/100.0}")


# import multiprocessing
#
# def worker(num,nam):
#     """thread worker function"""
#     print('Worker:', num)
#     return
#
# if __name__ == '__main__':
#     jobs = []
#     for i in range(4):
#         p = multiprocessing.Process(target=worker, args=(i,i+1,))
#         jobs.append(p)
#         p.start()