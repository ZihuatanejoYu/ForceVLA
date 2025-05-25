from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import ThisLaunchFileDir

def generate_launch_description():
    description_launch = IncludeLaunchDescription(
        [ThisLaunchFileDir(), '/display.launch.py']
    )

    rviz2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([ThisLaunchFileDir(), '/rviz.launch.py']),
    )

    return LaunchDescription([
        description_launch,
        rviz2_launch
    ])
