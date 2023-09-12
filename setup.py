from setuptools import setup

setup(
    name='Mario_GYM',
    version='1.0',
    description='Preprocessed Retro GYM for Super Mario Bros.',
    author='OAO',
    author_email='pjh9996@naver.com',
    packages=['mario_gym'],
    include_package_data=True,
    package_data={'': ['*.nes', ], },
    install_requires=[
        'numpy==1.19.4',
        'gym==0.17.3',
        'gym-retro==0.8.0',
        'PyQt6-sip==13.2.1',
        'PyQt6-Qt6==6.2.4',
        'PyQt6==6.2.3',
    ],
    python_requires='>=3.6, <3.7',
)
