from .hectnet import get_configuration

# Use the same configuration as the original HECTNet
# You can customize this if needed for the multiscale version
def get_configuration(opts):
    return get_configuration(opts)

from .ehfr_net import get_configuration

# Use the same configuration as the original EHFR_Net
# You can customize this if needed for the multiscale version
def get_configuration(opts):
    return get_configuration(opts)