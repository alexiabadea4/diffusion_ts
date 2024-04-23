import numpy as np

def makesignal(noise_level, signalclass):
    startoffset = 80  
    maxseglength = 180  
    minseglength = 80   
    
    def square(length, amp, signalclass):
        delta = float(signalclass) - 0.5
        return (amp + delta) * np.ones(length)
    
    def sine(length, amp, signalclass):
        return amp * np.sin(np.pi * np.linspace(0, 1, length))
    
    def ramp(length, amp, signalclass):
        return amp * np.linspace(0, 1, length)
    
    def cosine(length, amp, signalclass):
        delta = 0.25 * np.random.randn() + (float(signalclass) - 0.5)
        return amp * np.cos(np.pi * (1 + delta) * np.linspace(0, 1, length))
    
    def udramp(length, amp, signalclass):
        if (length % 2) == 1:
            length += 1
        a = np.linspace(0, amp, int(length / 2))
        b = np.linspace(amp, 0, int(length / 2))
        return np.concatenate((a, b))

    def getpattern(choice, seglength, signalclass):
        switcher = {
            1: square,
            2: sine,
            3: ramp,
            4: cosine,
            5: udramp
        }
        option = switcher.get(choice, lambda: np.zeros(seglength))
        segment = option(seglength, 2.0 + np.random.randn(), signalclass)
        return segment

    n = np.linspace(0, 1, 1024)  
    y = np.zeros_like(n)
    Choices = [1, 2, 3, 4, 5]
    
    patterntype = np.zeros((len(y), len(Choices) + 1))

    for _ in range(5):  
        seglength = np.round((maxseglength - minseglength) * np.random.rand()) + minseglength
        Choice = np.random.choice(Choices)
        seglength = int(seglength)
        segment = getpattern(Choice, seglength, signalclass)
        start_index = _ * maxseglength + startoffset
        end_index = start_index + len(segment)
        y[start_index:end_index] = segment
        patterntype[start_index:end_index, Choice] = 1
        Choices.remove(Choice)
    
    y = y + noise_level * np.random.randn(len(y))
    
    return y, patterntype, signalclass 

def generate_balanced_dataset_big(num_samples_per_class, noise_level):
    num_classes = 1
    num_samples = num_samples_per_class * num_classes
    signals = np.zeros((num_samples, 1024))
    masks = np.zeros((num_samples, 1024, 6))
    signalclasses = np.zeros(num_samples)
    
    for i in range(num_samples_per_class):
        for signalclass in range(num_classes):
            index = i + num_samples_per_class * signalclass
            s, gt, _ = makesignal(noise_level, signalclass)
            signals[index, :] = s
            masks[index, :, :] = gt
            signalclasses[index] = signalclass
    
    return signals, masks, signalclasses
