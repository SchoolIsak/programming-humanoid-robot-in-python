# NOTE: This file was edited 27.11, since it didn't work prior. This had to be corrected for the next assignment
'''In this exercise you need to implement inverse kinematics for NAO's legs

* Tasks:
    1. solve inverse kinematics for NAO's legs by using analytical or numerical method.
       You may need documentation of NAO's leg:
       http://doc.aldebaran.com/2-1/family/nao_h21/joints_h21.html
       http://doc.aldebaran.com/2-1/family/nao_h21/links_h21.html
    2. use the results of inverse kinematics to control NAO's legs (in InverseKinematicsAgent.set_transforms)
       and test your inverse kinematics implementation.
'''


from forward_kinematics import ForwardKinematicsAgent
from numpy.matlib import identity
import autograd.numpy as anp
from autograd import grad
import numpy as np


class InverseKinematicsAgent(ForwardKinematicsAgent):
    def inverse_kinematics(self, effector_name, transform):
        '''solve the inverse kinematics

        :param str effector_name: name of end effector, e.g. LLeg, RLeg
        :param transform: 4x4 transform matrix
        :return: list of joint angles
        '''

        # --- NAO Leg joint order ---
        leg_joints = [
            "LHipYawPitch" if effector_name == "LLeg" else "RHipYawPitch",
            "LHipRoll"     if effector_name == "LLeg" else "RHipRoll",
            "LHipPitch"    if effector_name == "LLeg" else "RHipPitch",
            "LKneePitch"   if effector_name == "LLeg" else "RKneePitch",
            "LAnklePitch"  if effector_name == "LLeg" else "RAnklePitch",
            "LAnkleRoll"   if effector_name == "LLeg" else "RAnkleRoll"
        ]

        # --- Full list of robot joints from ForwardKinematicsAgent ---
        all_joints = self.joint_names   # typically includes Head, Arms, Legs

        # Initial guess for optimized joints
        q0 = anp.zeros(len(leg_joints))

        T_target = anp.array(transform)

        # ---------- Loss function for autograd ----------
        def loss(q):
            # Build a *full* joint dictionary
            joint_dict = {}

            # Fill leg joints from q
            for i, joint in enumerate(leg_joints):
                joint_dict[joint] = q[i]

            # Fill all other joints with 0
            for j in all_joints:
                if j not in joint_dict:
                    joint_dict[j] = 0.0

            # Compute forward kinematics
            T_fk = self.forward_kinematics(joint_dict, effector=effector_name)

            pos_err = anp.sum((T_fk[:3, 3] - T_target[:3, 3]) ** 2)
            rot_err = anp.sum((T_fk[:3, :3] - T_target[:3, :3]) ** 2)

            return pos_err + rot_err

        dloss = grad(loss)

        # ---------- Gradient descent ----------
        q = q0.copy()
        lr = 0.02

        for _ in range(200):
            q = q - lr * dloss(q)

        return list(q)


    def set_transforms(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''

        joint_angles = self.inverse_kinematics(effector_name, transform)

        if effector_name == "LLeg":
            names = ["LHipYawPitch", "LHipRoll", "LHipPitch",
                     "LKneePitch", "LAnklePitch", "LAnkleRoll"]
        else:
            names = ["RHipYawPitch", "RHipRoll", "RHipPitch",
                     "RKneePitch", "RAnklePitch", "RAnkleRoll"]

        # keyframes = (list of joint names, list of angles, list of times)
        times = [[0.0, 1.0] for _ in joint_angles]

        # keys: for each joint we provide two angles [start_angle, end_angle].
        # Using the same angle twice will cause a constant target pose (no jump).
        # Convert to plain float to avoid numpy/autograd scalar types causing len()/type issues.
        keys = [[float(angle), float(angle)] for angle in joint_angles]

        # Proper keyframes format expected by angle_interpolation: (names, times, keys)
        self.keyframes = (names, times, keys)


if __name__ == '__main__':
    agent = InverseKinematicsAgent()
    # test inverse kinematics
    T = anp.eye(4)
    T[-1, 1] = 0.05
    T[-1, 2] = -0.26
    agent.set_transforms('LLeg', T)
    agent.run()


