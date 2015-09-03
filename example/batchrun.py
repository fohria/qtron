number_of_runs = 50

for x in xrange(0,number_of_runs):
    execfile("house_agent.py")

print "%s runs completed" % (number_of_runs)
