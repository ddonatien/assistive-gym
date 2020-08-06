from gym.envs.registration import register

tasks = ['ScratchItch', 'BedBathing', 'Feeding', 'Drinking', 'Dressing', 'ArmManipulation']
robots = ['PR2', 'Jaco', 'Baxter', 'Sawyer', 'Stretch']
human_states = ['', 'Human']

for task in tasks:
    for robot in robots:
        for human_state in human_states:
            register(
                id='%s%s%s-v1' % (task, robot, human_state),
                entry_point='assistive_gym.envs:%s%s%sEnv' % (task, robot, human_state),
                max_episode_steps=200,
            )

