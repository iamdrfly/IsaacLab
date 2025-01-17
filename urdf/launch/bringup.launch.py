# Copyright 2023 ros2_control Development Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from launch import LaunchDescription
from launch.actions import RegisterEventHandler
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.event_handlers import OnProcessExit
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory
from launch.launch_context import LaunchContext
from launch.conditions import IfCondition



def generate_launch_description():


    urdf = os.path.join(
        get_package_share_directory('grace_description'),
        'urdf',
        'grace_usd.urdf')

    with open(urdf, 'r') as infp:
        robot_desc = infp.read()

    uma = {'robot_description': robot_desc}

    robot_state_pub_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[uma],
    ) 

    rviz = Node(
            package='rviz2',
            namespace='',
            executable='rviz2',
            name='rviz2',
            arguments = ['-d' + os.path.join(get_package_share_directory('grace_description'), 'launch', 'config.rviz')]
        )

    nodes = [
        robot_state_pub_node,
        rviz
    ]


    return LaunchDescription(nodes)
