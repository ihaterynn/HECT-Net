from .hectnet import get_configuration as base_get_configuration

# Use the same configuration as the original HECTNet
# You can customize this if needed for the multiscale version
def get_configuration(opts):
    return base_get_configuration(opts)