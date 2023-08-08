import time
from IPython.display import clear_output
from matplotlib import pyplot as plt
import math
class progressbar:
    def __init__(self,timegap,length=40,char1='-',char2='>'):
        self.start_time = time.time()
        self.timegap=timegap
        self.time_ellapsed=0
        self.length = length
        self.char1 = char1
        self.char2 = char2
        self.data=[]
        self.extraword=''
        self.past = 0
        self.tim1 = 0
        self.past_error = 0
    def print(self,current,max,string='',clear=True,graph=False,error=0,graph_length=10,smoothing=0.2):
        assert smoothing<1
        progress=round(self.length*current/max)
        if progress == 0:
            self.past_error = error
        curr_time = time.time()
        if curr_time-self.start_time>self.timegap:
            self.time_spent = time.time()-self.start_time
            self.time_ellapsed+=self.timegap
            self.extraword+='\n Time taken ='+str(self.time_ellapsed)+'sec'
            if clear:
                clear_output(wait=True)
                if progress!=self.past:
                    self.tim1 = round((self.length-progress)/(progress/self.time_spent),2)
                print('['+self.char1*int(progress)+self.char2+' '*(self.length-progress)+']',
                        '\n',
                        string,
                        f"\ntime {self.tim1}s",
                        f"\ntime_spent {self.time_spent}s",
                        flush=True)
                self.past = progress
            if graph:
                if len(self.data)>graph_length:
                    self.data.pop(0)
                    # print(self.data,graph_length,len(self.data))
                error = smoothing*error+(1-smoothing)*self.past_error
                self.past_error = error
                self.data.append([self.time_ellapsed,error])
                plt.plot([x[0] for x in self.data],[x[1] for x in self.data])
                plt.show()