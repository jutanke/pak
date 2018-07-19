from distutils.core import setup

setup(
    name="pak",
    include_package_data=True,
    version="0.0.38",
    packages=[  "pak",
                "pak/util",
                "pak/datasets",
                "pak/evaluation",
                "pak/post_processing"],
)
