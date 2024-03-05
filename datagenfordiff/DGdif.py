import numpy as np

def makesignal(noise_level, signalclass):
# Returns a signal with some random
# structure as a 1x256 array, and
# a GT label (6x256) comprising 6 channels,
# one channel for each possible signal
# class. The Channels currently have values 
# of 0 if not that signal type, and 
# a value of 1 if during the indices that
# are between the start and end indices
# matching a particular pattern type
# In addition, there is a whole-signal
# class assigned to each full 256 length
# signal.


# Hard coded, suitable for length of 256
# samples; if longer, increase size
# proportionally for sensible dimensions
    startoffset=20
    maxseglength=45
    minseglength=20
    
    
    def square(length, amp, signalclass):
        delta = float(signalclass)-0.5
        return (amp+delta)*np.ones(length)
    
    def sine(length, amp, signalclass):
        return amp*np.sin(np.pi*np.linspace(0,1,length))

    def ramp(length, amp, signalclass):
        return amp*np.linspace(0, 1, length)

    def cosine(length, amp, signalclass):
        delta = 0.25*np.random.randn() + (float(signalclass)-0.5)
        return amp*np.cos(np.pi*(1+delta)*np.linspace(0,1,length))
    
    def udramp(length, amp, signalclass):
        if (length % 2) == 1:
            length += 1
    
        a=np.linspace(0, amp, int(length/2))
        b=np.linspace(amp, 0, int(length/2))
        
        return np.concatenate((a,b))

    def getpattern(choice, seglength, signalclass):
        switcher = {
         1: square,
         2: sine,
         3: ramp,
         4: cosine,
         5: udramp
        }
        option = switcher.get(choice, lambda: np.zeros(seglength))
        segment = option(seglength, 2.0+np.random.randn(), signalclass)
        
        return segment
    
    def pick(Choices):
    
        x = np.random.permutation(Choices)
        selected = x[0]
        
        return selected
    
    # The actual functions that define the signals
    n = np.linspace(0,1,256)
    y = np.zeros_like(n)
    Choices = [1,2,3,4,5]
    
    
    patterntype = np.zeros((len(y),len(Choices)+1))

    for n in range(5):
        seglength = np.round((maxseglength-minseglength)*np.random.rand())+minseglength
        Choice = pick(Choices)
        seglength = seglength.astype(int)
        segment = getpattern(Choice, seglength, signalclass)
        Choices.remove(Choice)
        
        y[n*maxseglength+startoffset:n*maxseglength+startoffset+len(segment)] = segment
        patterntype[n*maxseglength+startoffset:n*maxseglength+startoffset+seglength,Choice] = 1
    
    y = y + noise_level*np.random.randn(1,len(y))
        
    return y, patterntype, signalclass 

def generate_balanced_dataset(num_samples_per_class, noise_level):
    num_classes = 1  
    num_samples = num_samples_per_class * num_classes
    signals = np.zeros((num_samples, 256))
    masks = np.zeros((num_samples, 256, 6))
    signalclasses = np.zeros(num_samples)
    
    for i in range(num_samples_per_class):
        for signalclass in range(num_classes):
            index = i + num_samples_per_class * signalclass
            s, gt, _ = makesignal(noise_level, signalclass)
            signals[index, :] = s
            masks[index, :, :] = gt
            signalclasses[index] = signalclass
            
    return signals, masks, signalclasses
