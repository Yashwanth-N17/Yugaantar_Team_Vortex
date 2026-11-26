"""monsoon_r1 package for Multiverse of Monsoon."""

from .models.climate_input_encoder import (
    CalendarProjection,
    ClimateInputEncoder,
    ClimateProjection,
    GeographicProjection,
)
from .models.multi_scale_spatial_encoder import MultiScaleSpatialEncoder
from .models.temporal_dynamics_encoder import (
    ConvLSTM,
    ConvLSTMCell,
    TemporalDynamicsEncoder,
)
from .models.spatial_context_attention import (
    CBAM,
    ChannelAttention,
    SpatialAttention,
    SpatialContextAttention,
    SpatialContextAttentionModule,
)
from .models.forecast_decoder import (
    Block5ForecastDecoder,
    ClimateForecastDecoder,
    ForecastDecoder,
)
from .models.residual_state_transition import (
    Block6ResidualStateTransition,
    ClimateResidualStateTransition,
    ResidualStateTransition,
)
from .models.multiverse_model import (
    MonsoonArchitecture,
    MultiverseModel,
    MultiverseOfMonsoonModel,
)

__all__ = [
    "ClimateInputEncoder",
    "ClimateProjection",
    "GeographicProjection",
    "CalendarProjection",
    "MultiScaleSpatialEncoder",
    "ConvLSTMCell",
    "ConvLSTM",
    "TemporalDynamicsEncoder",
    "ChannelAttention",
    "SpatialAttention",
    "SpatialContextAttention",
    "SpatialContextAttentionModule",
    "CBAM",
    "ForecastDecoder",
    "ClimateForecastDecoder",
    "Block5ForecastDecoder",
    "ResidualStateTransition",
    "ClimateResidualStateTransition",
    "Block6ResidualStateTransition",
    "MultiverseOfMonsoonModel",
    "MultiverseModel",
    "MonsoonArchitecture",
]
