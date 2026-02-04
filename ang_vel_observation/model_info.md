
# Policy achitecture

		+---------------------------------------------------------+
		| Active Observation Terms in Group: 'policy' (shape: (48,)) |
		+-----------+---------------------------------+-----------+
		|   Index   | Name                            |   Shape   |
		+-----------+---------------------------------+-----------+
		|     0     | velocity_commands               |    (3,)   |
		|     1     | joint_pos                       |   (13,)   |
		|     2     | joint_vel                       |   (13,)   |
		|     3     | imu_ang_vel                     |    (3,)   |
		|     4     | actions                         |   (13,)   |
		+-----------+---------------------------------+-----------+
		
# train curriculum

* trained in rough terrain only
