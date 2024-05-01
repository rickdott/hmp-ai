from torchvision.transforms import Transform


class EegNoiseTransform(Transform):
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def __call__(self, segment):
        return segment


class EegTimeWarpingTransform(Transform):
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def __call__(self, segment):
        return segment


class EegChannelShufflingTransform(Transform):
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def __call__(self, segment):
        return segment
