from setuptools import find_packages, setup

package_name = 'intent_safety'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
package_data={
    'intent_safety': ['intent_model.pkl', 'vectorizer.pkl'],
},
include_package_data=True,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Mostofa Shakil Rafi',
    maintainer_email='rafi.2008044.ruet.mte@gmail.com',
    description='Intent classification and safety assessment components for robot task requests.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'intent_node = intent_safety.intent_node:main',
        ],
    },
)
