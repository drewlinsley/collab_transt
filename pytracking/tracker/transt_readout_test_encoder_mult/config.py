from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from yacs.config import CfgNode as CN

__C = CN()

cfg = __C

# ------------------------------------------------------------------------ #
# Tracker options
# ------------------------------------------------------------------------ #
__C.TRACK = CN()

__C.TRACK.TYPE = ''

# Scale penalty
__C.TRACK.PENALTY_K = 0

# Window influence
__C.TRACK.WINDOW_INFLUENCE = 0.60

# Exemplar size
__C.TRACK.EXEMPLAR_SIZE = 64

# Instance size
__C.TRACK.INSTANCE_SIZE = 96

