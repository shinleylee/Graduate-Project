print(str(   ("%.100f" % ((1/81)*10))))

for i in range(1, 81):
    if '2018' in str(   ("%.100f" % (i/81))):
        print(i)
        print(str(   ("%.100f" % (i/81))))