import time


start = time.time()
for i in range(10000):
    for j in range(10000):
        a=j

print("time", time.time()-start)