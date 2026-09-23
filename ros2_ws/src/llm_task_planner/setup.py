from setuptools import find_packages, setup

package_name = 'llm_task_planner'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mostofa Shakil Rafi',
    maintainer_email='rafi.2008044.ruet.mte@gmail.com',
    description='LLM-based natural language task planning for the trustworthy service robot framework.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "llm_task_planner = llm_task_planner.llm_task_planner:main",
        ],
    },
)
