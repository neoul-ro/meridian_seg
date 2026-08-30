from glob import glob

from setuptools import find_packages, setup

package_name = 'meridian_seg'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        # weights/는 일부러 설치하지 않는다. .engine 하나가 26MB~142MB인데
        # colcon은 이것을 install 트리로 복사한다. 노드는 소스 트리에서 직접
        # 찾으므로(--symlink-install) 그대로 동작하고, 아니면 model_path
        # 파라미터나 MERIDIAN_SEG_ENGINE 환경변수로 지정하면 된다.
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='blu-y',
    maintainer_email='a_o@kakao.com',
    description='FastSAM segmentation node for the Meridian perception pipeline.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'seg_node = meridian_seg.seg_node:main',
            'geobuilder_node = meridian_seg.geobuilder_node:main',
        ],
    },
)
