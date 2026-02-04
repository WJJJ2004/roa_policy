
# Policy achitecture

		+---------------------------------------------------------+
		| Active Observation Terms in Group: 'policy' (shape: (48,)) |
		+-----------+---------------------------------+-----------+
		|   Index   | Name                            |   Shape   |
		+-----------+---------------------------------+-----------+
		|     0     | projected_gravity               |    (3,)   |
		|     1     | velocity_commands               |    (3,)   |
		|     2     | joint_pos                       |   (13,)   |
		|     3     | joint_vel                       |   (13,)   |
		|     4     | imu_ang_vel                     |    (3,)   |
		|     5     | actions                         |   (13,)   |
		+-----------+---------------------------------+-----------+
		
# train curriculum

* trained in rough terrain >> retrain in flat 


plz donot open
