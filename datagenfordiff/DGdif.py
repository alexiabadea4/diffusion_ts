

import numpy as np

def makesignal(noise_level, signalclass):
    startoffset = 20
    maxseglength = 45
    minseglength = 20
    maxzero = 40  # Maximum length of zero segments
    minzero = 1  # Minimum length of zero segments
    
    # Define signal patterns
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
        if length % 2 == 1:
            length += 1
        a = np.linspace(0, amp, int(length / 2))
        b = np.linspace(amp, 0, int(length / 2))
        return np.concatenate((a, b))

    def getpattern(choice, seglength, signalclass):
        switcher = {1: square, 2: sine, 3: ramp, 4: cosine, 5: udramp}
        option = switcher.get(choice, lambda: np.zeros(seglength))
        segment = option(seglength, 2.0 + np.random.randn(), signalclass)
        return segment

    def pick(Choices):
        x = np.random.permutation(Choices)
        selected = x[0]
        return selected

    # Initialize signal and pattern type
    n = np.linspace(0, 1, 256)
    y = np.zeros_like(n)
    Choices = [1, 2, 3, 4, 5]
    patterntype = np.zeros((len(y), len(Choices) + 1))
    current_pos = startoffset

    # Generate signal with variable zero lengths
    while current_pos < len(y) - maxseglength and Choices:
        seglength = np.round((maxseglength - minseglength) * np.random.rand()) + minseglength
        Choice = pick(Choices)
        seglength = int(seglength)
        segment = getpattern(Choice, seglength, signalclass)
        zerolength = np.random.randint(minzero, maxzero + 1)
        y[current_pos:current_pos + len(segment)] = segment
        patterntype[current_pos:current_pos + seglength, Choice] = 1
        current_pos += seglength + zerolength
        Choices.remove(Choice)

    y += noise_level * np.random.randn(len(y))

    return y, patterntype, signalclass

def generate_balanced_dataset(num_samples_per_class, noise_level):
    num_classes = 5
    num_samples = num_samples_per_class * num_classes
    signals = np.zeros((num_samples, 256))
    masks = np.zeros((num_samples, 256, 6))
    signalclasses = np.zeros(num_samples)

    for signalclass in range(num_classes):
        for i in range(num_samples_per_class):
            index = i + num_samples_per_class * signalclass
            s, gt, _ = makesignal(noise_level, signalclass)
            signals[index, :] = s
            masks[index, :, :] = gt
            signalclasses[index] = signalclass

    return signals, masks, signalclasses