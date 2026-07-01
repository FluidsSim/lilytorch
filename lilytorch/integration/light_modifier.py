"""TaskExtension that patches MuJoCo scene lighting to reduce reflections.

The default FARMS light has specular=0.7, which causes bright mirror-like
highlights on wet/shiny surfaces.  This extension runs once at episode start
and overwrites the per-light colour arrays on the MjModel, replacing them with
a soft, specular-free setup that is suitable for tank recordings.

It also deactivates the camera-mounted headlight (``vis.headlight``), which
fires at whatever the camera is looking at and adds another source of harsh
reflections.
"""

from farms_core.simulation.extensions import TaskExtension


class LightModifier(TaskExtension):
    """Patch MuJoCo scene lights and headlight to reduce specular reflections.

    Parameters
    ----------
    specular : list[float] | None
        RGB specular colour for all scene lights.  Defaults to [0, 0, 0]
        (no specular highlight).
    diffuse : list[float] | None
        RGB diffuse colour.  Defaults to [0.45, 0.45, 0.45].
    ambient : list[float] | None
        RGB ambient colour.  Defaults to [0.55, 0.55, 0.55].
    disable_headlight : bool
        If True (default) the camera-mounted headlight is switched off
        entirely.  If False its specular component is zeroed instead.
    """

    def __init__(
        self,
        *,
        specular=None,
        diffuse=None,
        ambient=None,
        disable_headlight: bool = True,
    ):
        super().__init__()
        self.specular = list(specular) if specular is not None else [0.0, 0.0, 0.0]
        self.diffuse  = list(diffuse)  if diffuse  is not None else [0.45, 0.45, 0.45]
        self.ambient  = list(ambient)  if ambient  is not None else [0.55, 0.55, 0.55]
        self.disable_headlight = disable_headlight

    @classmethod
    def from_options(cls, config: dict, experiment_options):
        del experiment_options
        return cls(
            specular          = config.get('specular'),
            diffuse           = config.get('diffuse'),
            ambient           = config.get('ambient'),
            disable_headlight = config.get('disable_headlight', True),
        )

    def initialize_episode(self, task, physics):
        del task
        model = physics.model
        # Prefer the raw mujoco.MjModel ptr so we can index light arrays
        # directly; fall back to the dm_control wrapper if ptr is absent.
        ptr = getattr(model, 'ptr', model)
        if ptr.nlight > 0:
            ptr.light_specular[:] = self.specular
            ptr.light_diffuse[:]  = self.diffuse
            ptr.light_ambient[:]  = self.ambient
        # Camera-mounted headlight
        hl = model.vis.headlight
        if self.disable_headlight:
            hl.active = 0
        else:
            hl.specular[:] = self.specular
