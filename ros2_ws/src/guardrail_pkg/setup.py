from setuptools import setup

package_name = 'guardrail_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='yourname',
    maintainer_email='your@email.com',
    description='Guardrail and secure comms nodes',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'guardrail_node = guardrail_pkg.guardrail_node:main',
            'secure_pub = guardrail_pkg.secure_pub:main',
            'secure_sub = guardrail_pkg.secure_sub:main',
            'otp_auth_node = guardrail_pkg.otp_auth_node:main',
            'robot_cli = guardrail_pkg.robot_cli:main',
            'local_ml_guardrail = guardrail_pkg.local_ml_guardrail:main',
            'clarification_node = guardrail_pkg.clarification_node:main',
            'robot_controller = guardrail_pkg.robot_controller:main',
            'yolo_detector = guardrail_pkg.yolo_detector:main',
            'add_collision_object = guardrail_pkg.add_collision_object:main',
            'hdre_node = guardrail_pkg.hdre_node:main',
            'context_verification_node = guardrail_pkg.context_verification_node:main',

        ],
    },
)
