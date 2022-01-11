import time
t1 = time.time()

for k in range(1000000):
    pass

t2 = time.time()
t = t2-t1
print("%.20f" % t)