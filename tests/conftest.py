import warnings

warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    module=r".*paddlex.*",
)
