from setuptools import Extension, setup


# Required dependencies.
install_requires = [
    "cuda_python",
    "netCDF4",
    "numpy",
    "scipy",
    "sqlalchemy",
    "xarray",
]


# Documentation dependencies.
doc_requires = [
    "sphinx",
    "sphinxcontrib-apidoc",
    "sphinxcontrib-napoleon",
    "sphinx-autopackagesummary",
    "sphinx_pangeo_theme",
]


# Optional dependencies.
extras_require = {
    "complete": install_requires,
    "docs": doc_requires,
    "arts": ["pyarts",]
}


# Entry points.
entry_points = {
   "pyLBL": ["Gas=pyLBL.c_lib.gas_optics:Gas",],
   "mt_ckd": [
       "CO2Continuum=pyLBL.mt_ckd.carbon_dioxide:CarbonDioxideContinuum",
       "H2OForeignContinuum=pyLBL.mt_ckd.water_vapor:WaterVaporForeignContinuum",
       "H2OSelfContinuum=pyLBL.mt_ckd.water_vapor:WaterVaporSelfContinuum",
       "N2Continuum=pyLBL.mt_ckd.nitrogen:NitrogenContinuum",
       "O2Continuum=pyLBL.mt_ckd.oxygen:OxygenContinuum",
       "O3Continuum=pyLBL.mt_ckd.ozone:OzoneContinuum",
   ],
   "arts_crossfit": ["CrossSection=pyLBL.arts_crossfit.cross_section:CrossSection"],
   "arts": ["Gas=pyLBL.pyarts_frontend.frontend:PyArtsGas",],
}


setup(
    name="pyLBL",
    version="0.0.1",
    description="Line-by-line absorption calculators.",
    url="https://github.com/GRIPS-code/pyLBL",
    author="pyLBL Developers",
    license="LGPL-2.1",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v2 or later (LGPLv2+)",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.6",
    packages=[
        "pyLBL",
        "pyLBL.c_lib",
        "pyLBL.webapi",
        "pyLBL.pyarts_frontend",
        "pyLBL.mt_ckd",
        "pyLBL.arts_crossfit",
    ],
    install_requires=install_requires,
    extras_require=extras_require,
    entry_points=entry_points,
    package_data={"": ["*.nc"], },
)
