# Models module
from .specmamba_net import (
    SpecMambaNet, SpectralBlock, PseudoMambaBlock, SpecMambaBlock,
    specmamba_small, specmamba_base, specmamba_large,
)
from .hrnet_dcn import HRNetDCN, HRNetStem, Bottleneck, FuseLayer  # deprecated
from .pcshear_hrnet import (
    PCShearHRNet,
    pcshear_hrnet_small,
    pcshear_hrnet_base,
    pcshear_hrnet_spectral,
)

__all__ = [
    # Primary architecture
    'SpecMambaNet', 'SpectralBlock', 'PseudoMambaBlock', 'SpecMambaBlock',
    'specmamba_small', 'specmamba_base', 'specmamba_large',
    # Legacy (deprecated)
    'HRNetDCN', 'HRNetStem', 'Bottleneck', 'FuseLayer',
    'PCShearHRNet', 'pcshear_hrnet_small', 'pcshear_hrnet_base',
    'pcshear_hrnet_spectral',
]
