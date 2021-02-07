#!/usr/bin/python3

# import _thread
import threading
import time


# Define a function for the thread
def print_time(thread_name, delay):
    count = 0
    while count < 5:
        time.sleep(delay)
        count += 1
        print("%s: %s" % (thread_name, time.ctime(time.time())))


# Create two threads as follows
NUM_THREADS = 2
threads = []
try:
    for i in range(NUM_THREADS):
        print("In main: creating thread {}".format(i))
        x = threading.Thread(target=print_time, args=('Thread-{}'.format(i), 2 * (i + 1)))
        threads.append(x)
        x.start()
except:
    print("Error: unable to create thread")

try:
    for thread in threads:
        thread.join()
except:
    print("Error: unable to start thread")

print("In main: All threads completed successfully")
