

@dataclass
class PeftConfig:
    """Holds the configuration for parameter-efficient finetuning."""

    method: Optional[str] = None
    """The PEFT method to train with."""


def peft_config() -> PeftConfig:
    """Return a runtime-generated :class:`PeftConfig` instance."""
    field_names = tuple(_methods.keys())

    # Override the `method` field with stricter `Literal[methods...]` type hint.
    fields: List[Tuple[str, Any, Any]] = [
        ("method", Optional[Literal[field_names]], field(default=None)),
    ]

    # Add a configuration field for each PEFT method.
    for name, kls in _methods.items():
        fields.append((name, kls, field(default_factory=kls)))

    config_kls = make_dataclass("_DynamicPeftConfig", fields, bases=(PeftConfig,))

    return cast(PeftConfig, config_kls())


def maybe_apply_peft(model: Module, config: PeftConfig) -> Module:
    if config.method is None:
        return model

    try:
        method_config = getattr(config, config.method)
    except AttributeError:
        method_config = None

    if not is_dataclass_instance(method_config):
        raise RuntimeError("`method` is not a valid PEFT method name.")

    try:
        method_fn = _method_functions[config.method]
    except KeyError:
        raise RuntimeError("report") from None

    return method_fn(model, method_config)


_methods: Dict[str, Type[DataClass]] = {}

_method_functions: Dict[str, Callable[[Module, Any], Module]] = {}


PeftMethodConfigT = TypeVar("PeftMethodConfigT", bound=DataClass)

def register_peft_method(name: str, config_kls: Type[DataClass]
