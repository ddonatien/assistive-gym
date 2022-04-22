from .dressingStanding import DressingStEnv
from .agents import pr2, baxter, sawyer, jaco, stretch, panda, human
from .agents.pr2 import PR2
from .agents.baxter import Baxter
from .agents.sawyer import Sawyer
from .agents.jaco import Jaco
from .agents.stretch import Stretch
from .agents.panda import Panda
from .agents.human import Human
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.tune.registry import register_env

robot_arm = 'left'
human_controllable_joint_indices = human.left_arm_joints
class DressingStPR2Env(DressingStEnv):
    def __init__(self):
        super(DressingStPR2Env, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DressingStBaxterEnv(DressingStEnv):
    def __init__(self):
        super(DressingStBaxterEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DressingStSawyerEnv(DressingStEnv):
    def __init__(self):
        super(DressingStSawyerEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))
register_env('assistive_gym:DressingStSawyer-v1', lambda config: DressingStSawyerEnv())

class DressingStJacoEnv(DressingStEnv):
    def __init__(self):
        super(DressingStJacoEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DressingStStretchEnv(DressingStEnv):
    def __init__(self):
        super(DressingStStretchEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DressingStPandaEnv(DressingStEnv):
    def __init__(self):
        super(DressingStPandaEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))

class DressingStPR2HumanEnv(DressingStEnv, MultiAgentEnv):
    def __init__(self):
        super(DressingStPR2HumanEnv, self).__init__(robot=PR2(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:DressingStPR2Human-v1', lambda config: DressingStPR2HumanEnv())

class DressingStBaxterHumanEnv(DressingStEnv, MultiAgentEnv):
    def __init__(self):
        super(DressingStBaxterHumanEnv, self).__init__(robot=Baxter(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:DressingStBaxterHuman-v1', lambda config: DressingStBaxterHumanEnv())

class DressingStSawyerHumanEnv(DressingStEnv, MultiAgentEnv):
    def __init__(self):
        super(DressingStSawyerHumanEnv, self).__init__(robot=Sawyer(robot_arm), human=Human(human_controllable_joint_indices, controllable=False))
register_env('assistive_gym:DressingStSawyerHuman-v1', lambda config: DressingStSawyerHumanEnv())

class DressingStJacoHumanEnv(DressingStEnv, MultiAgentEnv):
    def __init__(self):
        super(DressingStJacoHumanEnv, self).__init__(robot=Jaco(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:DressingStJacoHuman-v1', lambda config: DressingStJacoHumanEnv())

class DressingStStretchHumanEnv(DressingStEnv, MultiAgentEnv):
    def __init__(self):
        super(DressingStStretchHumanEnv, self).__init__(robot=Stretch('wheel_'+robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:DressingStStretchHuman-v1', lambda config: DressingStStretchHumanEnv())

class DressingStPandaHumanEnv(DressingStEnv, MultiAgentEnv):
    def __init__(self):
        super(DressingStPandaHumanEnv, self).__init__(robot=Panda(robot_arm), human=Human(human_controllable_joint_indices, controllable=True))
register_env('assistive_gym:DressingStPandaHuman-v1', lambda config: DressingStPandaHumanEnv())

