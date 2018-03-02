import numpy as np

class LinearSchedule(object):
    def __init__(self, begin, end, nsteps):
        self.val        = begin
        self.begin      = begin
        self.end        = end
        self.nsteps     = nsteps


    def update(self, t):
        if t==0:
            self.val=self.begin
        elif t==self.nsteps:
            self.val=self.end
        else:
            self.val=self.begin+ float(t)/float(self.nsteps)*(self.end-self.begin)
        if self.val<self.end:
            self.val=self.end



if __name__ == "__main__":
    pass