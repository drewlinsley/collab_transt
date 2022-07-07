from pytracking.utils import TrackerParams
from pytracking.features.net_wrappers import NetWithBackbone

def parameters():
    params = TrackerParams()
    params.debug = 0
    params.visualization = False
    params.use_gpu = True
    # params.net = NetWithBackbone(net_path='transt.pth',
    #                              use_gpu=params.use_gpu)
    params.net = NetWithBackbone(net_path='/content/TransT/pytracking/networks/ckpt.pth.tar',
                                 use_gpu=params.use_gpu)
    return params
