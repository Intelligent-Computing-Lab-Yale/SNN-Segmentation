class AverageMeterNetwork(object):
    """
    Computes and stores lists of values for the network
    """

    def __init__(self, length):
        self.length = length
        self.reset()

    def reset(self):
        self.sum = [0] * self.length
        self.count = 0
        self.units = [0] * self.length

    def updateSum(self, index, val):
        self.sum[index] += val

    def updateCount(self, n):
        self.count += n
    
    def average(self):
        return [s / self.count for s in self.sum]

    def totalAverage(self):
        return sum(self.sum) / self.count

    def updateUnits(self, index, u):
        self.units[index] += u

    def rate(self):
        return [i / j for i, j in zip(self.average(), self.units)]

    def totalRate(self):
        return sum(self.average()) / self.totalUnits()

    def totalUnits(self):
        return sum(self.units)