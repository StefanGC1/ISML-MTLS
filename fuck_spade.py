import spade, sys, pathlib

print("\n".join(
    str(pathlib.Path(p) / "spade")          # potential culprit
    for p in sys.path
    if (pathlib.Path(p) / "spade").exists()
))

print(spade)                 # shows the module object
print(spade.__file__)        # where it was loaded from
print(pathlib.Path(spade.__file__).parent)  # the package folder