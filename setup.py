from setuptools import setup, PEP420PackageFinder

setup(
    name="cascade_at",
    version='0.0.1',
    description='DisMod AT Wrapper',
    package_dir={"": "src"},
    packages=PEP420PackageFinder.find("src"),
    install_requires=[
        "numpy==1.17.2",
        "pandas==0.25.1",
        "scipy",
        "sqlalchemy",
        "dill",
        "intervaltree",
        "tables",
        "networkx"
    ],
    zip_safe=False,
    extras_require={
        "ihme": [
            "db-tools==0.9.0",
            "db-queries==21.0.0",
            "gbd==2.1.0",
            "jobmon==1.1.1",
            "elmo==1.6.18",
            "ihme-rules==2.1.0"],
        "testing": [
            "hypothesis", "pytest", "pytest-mock"
        ]
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Licence :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Statistics"
    ],
    entry_points={'console_scripts': [
        'configure_inputs=cascade_at.executor.configure_inputs:main',
        'dismod_db=cascade_at.executor.dismod_db:main',
        'sample_simulate=cascade_at.executor.sample_simulate:main',
        'format_upload=cascade_at.executor.format_upload:main',
        'cleanup=cascade_at.executor.cleanup:main',
        'run_cascade=cascade_at.executor.run:main',
        'run_dmdismod=cascade_at.executor.run_dmdismod:main'
    ]}
)
