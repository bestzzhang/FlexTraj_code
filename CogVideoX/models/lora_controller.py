from peft.tuners.tuners_utils import BaseTunerLayer
from typing import List, Any, Optional, Type


class enable_lora:
    def __init__(self, lora_modules: List[BaseTunerLayer], activated: bool) -> None:
        self.activated: bool = activated
        if activated:
            return
        self.lora_modules: List[BaseTunerLayer] = [
            each for each in lora_modules if isinstance(each, BaseTunerLayer)
        ]
        self.scales = [
            {
                active_adapter: lora_module.scaling[active_adapter]
                for active_adapter in lora_module.active_adapters
            }
            for lora_module in self.lora_modules
        ]

    def __enter__(self) -> None:
        if self.activated:
            return

        for lora_module in self.lora_modules:
            if not isinstance(lora_module, BaseTunerLayer):
                print(f"{lora_module} is not BaseTunerLayer")
                continue
            lora_module.scale_layer(0)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        if self.activated:
            return
        for i, lora_module in enumerate(self.lora_modules):
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            for active_adapter in lora_module.active_adapters:
                lora_module.scaling[active_adapter] = self.scales[i][active_adapter]

''' TODO: modify this code to enable training-free multi-condition
class enable_lora:
    def __init__(
        self,
        lora_modules: List[BaseTunerLayer],
        activated: bool = False,
        disable: Optional[Union[str, Iterable[str]]] = None,
    ):
        self.activated = activated
        if activated:
            return                         # fast-exit, nothing to store

        # normalise the disable list
        if disable is None:
            self.disable = {"*"}           # wildcard → disable all
        elif isinstance(disable, str):
            self.disable = {disable}
        else:
            self.disable = set(disable)

        # keep references and original scales
        self.lora_modules = [
            m for m in lora_modules if isinstance(m, BaseTunerLayer)
        ]
        self.saved_scales = [
            {name: m.scaling[name] for name in m.active_adapters}
            for m in self.lora_modules
        ]

    def __enter__(self):
        if self.activated:
            return

        for m in self.lora_modules:
            for name in m.active_adapters:
                if "*" in self.disable or name in self.disable:
                    m.scaling[name] = 0.0  # mute
'''


class set_lora_scale:
    def __init__(self, lora_modules: List[BaseTunerLayer], scale: float) -> None:
        self.lora_modules: List[BaseTunerLayer] = [
            each for each in lora_modules if isinstance(each, BaseTunerLayer)
        ]
        self.scales = [
            {
                active_adapter: lora_module.scaling[active_adapter]
                for active_adapter in lora_module.active_adapters
            }
            for lora_module in self.lora_modules
        ]
        self.scale = scale

    def __enter__(self) -> None:
        for lora_module in self.lora_modules:
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            lora_module.scale_layer(self.scale)

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[Any],
    ) -> None:
        for i, lora_module in enumerate(self.lora_modules):
            if not isinstance(lora_module, BaseTunerLayer):
                continue
            for active_adapter in lora_module.active_adapters:
                lora_module.scaling[active_adapter] = self.scales[i][active_adapter]
