import time

class TimeTracker:
    def __init__(self, name):
        self.total_time = 0
        self.name = name

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, exc_type, exc_value, traceback):
        self.total_time += time.time()-self.start_time

class Profiler:
    def __init__(self):
        self.timer = {}

    def getProfiler(self, name):
        if name not in self.timer:
            self.timer[name] = TimeTracker(name)
        return self.timer[name]

    def getTime(self, name):
        return self.timer[name].total_time

    def printStats(self):
        for name in self.timer:
            print(name, ":", self.timer[name].total_time)

profiler = Profiler()