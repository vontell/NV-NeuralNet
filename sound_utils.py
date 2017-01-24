import pyaudio
import numpy
import math
import threading
import Queue

class Sounds(object):
    global RATE
    RATE = 44100
    
    def __init__(self):
        self.p = None
        self.stream = None
        self.thr = None
        self.queue = Queue.Queue()
    
    def open(self):
        if self.thr is None or not self.thr.is_alive():
            self.p = pyaudio.PyAudio()
            self.stream = self.p.open(format=pyaudio.paFloat32,
                                 channels=1,
                                 rate=RATE,
                                 output=1)
            def run():
                while True:
                    elem = self.queue.get()
                    if elem is None:
                        break
                    freq, dur = elem
                    self.play_tone(freq, dur)
            self.thr = threading.Thread(target=run, args=(), kwargs={})
            self.thr.daemon = True
            self.thr.start()
    
    def close(self):
        try:
            self.queue.put(None)
            self.thr.join()
        except:
            pass
        self.queue = Queue.Queue()
        self.stream.close()
        self.p.terminate()
        
    def sine(self, frequency, length, rate):
        length = int(length * rate)
        factor = float(frequency) * (math.pi * 2) / rate
        return numpy.sin(numpy.arange(length) * factor)

    def play_tone(self, frequency=440, length=1):
        chunks = []
        chunks.append(self.sine(frequency, length, RATE))
        chunk = numpy.concatenate(chunks) * 0.25
        self.stream.write(chunk.astype(numpy.float32).tostring())
    
    def queue_play(self, tup):
        self.queue.put(tup)
        #self.queue.put((0, 0.05)) # Add delay
    
    def alert(self):
        self.queue_play((440, 0.2))
    
    def success(self):
        self.queue_play((783.991, 0.1))


if __name__ == '__main__':
    sounds = Sounds()
    sounds.open()
    
    print 'a'
    
    sounds.alert()
    sounds.success()
    sounds.alert()
    sounds.success()
    sounds.alert()
    sounds.success()
    sounds.alert()
    
    print 'b'
    
    sounds.close()
    