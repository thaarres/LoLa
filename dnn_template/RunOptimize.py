import time
from redis import Redis
from rq import Queue

from TrainClassifiers import main

q = Queue('high', connection=Redis(host='daint101', port=23836))

job = q.enqueue(main, timeout=-1)

while True:
    print(job.result)
    time.sleep(10)

