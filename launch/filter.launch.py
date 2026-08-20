from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    package_share = get_package_share_directory('pusher_ghost_filter')
    config = LaunchConfiguration('config')

    return LaunchDescription([
        DeclareLaunchArgument(
            'config', default_value=package_share + '/config/default.yaml'),
        Node(
            package='pusher_ghost_filter',
            executable='pusher_ghost_filter_node',
            name='pusher_ghost_filter',
            output='screen',
            parameters=[config],
        ),
    ])
