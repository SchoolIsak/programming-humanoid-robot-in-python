'''In this exercise you need to implement forward kinematics for NAO robot

* Tasks:
    1. complete the kinematics chain definition (self.chains in class ForwardKinematicsAgent)
       The documentation from Aldebaran is here:
       http://doc.aldebaran.com/2-1/family/robots/bodyparts.html#effector-chain
    2. implement the calculation of local transformation for one joint in function
       ForwardKinematicsAgent.local_trans. The necessary documentation are:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    3. complete function ForwardKinematicsAgent.forward_kinematics, save the transforms of all body parts in torso
       coordinate into self.transforms of class ForwardKinematicsAgent

* Hints:
    1. the local_trans has to consider different joint axes and link parameters for different joints
    2. Please use radians and meters as unit.
'''

# add PYTHONPATH
import os
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'joint_control'))

import autograd.numpy as anp


from recognize_posture import PostureRecognitionAgent


class ForwardKinematicsAgent(PostureRecognitionAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(ForwardKinematicsAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.transforms = {n: anp.eye(4) for n in self.joint_names}

        # chains defines the name of chain and joints of the chain
        self.chains = {'Head': ['HeadYaw', 'HeadPitch'],
                       'LArm': ['LShoulderPitch', 'LShoulderRoll', 'LElbowYaw', 'LElbowRoll'], #, 'LWristYaw','LHand'
                       'RArm': ['RShoulderPitch', 'RShoulderRoll', 'RElbowYaw', 'RElbowRoll'], #, 'RWristYaw','RHand'
                       'LLeg': ['LHipYawPitch', 'LHipRoll', 'LHipPitch', 'LKneePitch', 'LAnklePitch', 'LAnkleRoll'],
                       'RLeg': ['RHipYawPitch', 'RHipRoll', 'RHipPitch', 'RKneePitch', 'RAnklePitch', 'RAnkleRoll']
                       }

    def think(self, perception):
        self.forward_kinematics(perception.joint)
        return super(ForwardKinematicsAgent, self).think(perception)

    def local_trans(self, joint_name, angle):
        T = anp.eye(4)

       
        # First we consider the rotation axis for every joint
        if joint_name in ["HeadYaw", "LElbowYaw", "RElbowYaw"]:
            # Z axis
            R = anp.array([[anp.cos(angle), -anp.sin(angle), 0],
                           [anp.sin(angle),  anp.cos(angle), 0],
                           [0, 0, 1]]) 

        elif joint_name in ["HeadPitch", "LShoulderPitch", "RShoulderPitch",
                            "LHipPitch", "RHipPitch",
                            "LKneePitch", "RKneePitch",
                            "LAnklePitch", "RAnklePitch"]:
            # Y axis
            R = anp.array([[anp.cos(angle), 0, anp.sin(angle)],
                           [0, 1, 0],
                           [-anp.sin(angle), 0, anp.cos(angle)]]) 

        elif joint_name in ["LShoulderRoll", "RShoulderRoll",
                            "LElbowRoll", "RElbowRoll",
                            "LHipRoll", "RHipRoll",
                            "LAnkleRoll", "RAnkleRoll"]:
            # X axis
            R = anp.array([[1, 0, 0],
                           [0, anp.cos(angle), -anp.sin(angle)],
                           [0, anp.sin(angle),  anp.cos(angle)]])  # 

        # Special: HipYawPitch has a 45° rotated axis
        elif joint_name in ["LHipYawPitch", "RHipYawPitch"]:
            c = anp.cos(angle)
            s = anp.sin(angle)
            axis = 1.0 / anp.sqrt(2.0)
            # Rotation around axis (x+z)/√2
            R = anp.array([
                [axis*axis*(1-c)+c, 0, axis*axis*(1-c)-axis*s],
                [0, 1, 0],
                [axis*axis*(1-c)+axis*s, 0, axis*axis*(1-c)+c]
            ])
        else:
            # Else we set R to be eye so no error
            R = anp.eye(3)

       # Joint translations
        dx = dy = dz = 0.0

        # We get the needed values directly from the documentation
        translations = {
            # Head
            "HeadYaw":         (0, 0, 0.1265),
            "HeadPitch":       (0, 0, 0),

            # Left arm
            "LShoulderPitch":  (0.0, +0.098, +0.100),
            "LShoulderRoll":   (0, 0, 0),
            "LElbowYaw":       (0.105, 0.015, 0),
            "LElbowRoll":      (0.0, 0, 0),

            # Right arm
            "RShoulderPitch":  (0.0, -0.098, +0.100),
            "RShoulderRoll":   (0, 0, 0),
            "RElbowYaw":       (0.105, -0.015, 0),
            "RElbowRoll":      (0, 0, 0),

            # Left leg
            "LHipYawPitch":    (0, +0.05, -0.085),
            "LHipRoll":        (0, 0, 0),
            "LHipPitch":       (0, 0, -0.10),
            "LKneePitch":      (0, 0, -0.1029),
            "LAnklePitch":     (0, 0, -0.1029),

            # Right leg
            "RHipYawPitch":    (0, -0.05, -0.085),
            "RHipRoll":        (0, 0, 0),
            "RHipPitch":       (0, 0, -0.10),
            "RKneePitch":      (0, 0, -0.1029),
            "RAnklePitch":     (0, 0, -0.1029),
        }

        if joint_name in translations:
            dx, dy, dz = translations[joint_name]

       # Construct output transformation matrix
        T = T.copy()  # important for autograd
        T[0:3, 0:3] = R
        T[0:3, 3] = anp.array([dx, dy, dz])

        return T

    def forward_kinematics(self, joints):
        for chain, chain_joints in self.chains.items():
            T = anp.eye(4)
            for j in chain_joints:
                angle = joints[j]
                T = T @ self.local_trans(j, angle)
                self.transforms[j] = T

    

if __name__ == '__main__':
    agent = ForwardKinematicsAgent()
    agent.run()
