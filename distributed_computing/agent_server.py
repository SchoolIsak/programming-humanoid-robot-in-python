'''In this file you need to implement remote procedure call (RPC) server

* There are different RPC libraries for python, such as xmlrpclib, json-rpc. You are free to choose.
* The following functions have to be implemented and exported:
 * get_angle
 * set_angle
 * get_posture
 * execute_keyframes
 * get_transform
 * set_transform
* You can test RPC server with ipython before implementing agent_client.py
'''

import os
import sys
from xmlrpc.server import SimpleXMLRPCServer
sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', 'kinematics'))

from inverse_kinematics import InverseKinematicsAgent

class ServerAgent(InverseKinematicsAgent):
    '''ServerAgent provides RPC service
    '''
    
    def get_angle(self, joint_name):
        '''get sensor value of given joint'''
        return self.perception.joint.get(joint_name)

    def set_angle(self, joint_name, angle):
        '''set target angle of joint for PID controller'''
        self.target_joints[joint_name] = angle
        return True

    def get_posture(self):
        '''return current posture of robot'''
        features = [
            self.perception.joint['LHipYawPitch'],
            self.perception.joint['LHipRoll'],
            self.perception.joint['LHipPitch'],
            self.perception.joint['LKneePitch'],
            self.perception.joint['RHipYawPitch'],
            self.perception.joint['RHipRoll'],
            self.perception.joint['RHipPitch'],
            self.perception.joint['RKneePitch'],
            self.perception.imu[0],
            self.perception.imu[1]
        ]
        # Reshaping ?
        features = [features]  # shape (1, 10)

        # Posture index prediction
        posture = self.posture_classifier.predict(features)[0]

        # print(posture)

        return str(posture)
       

    def execute_keyframes(self, keyframes):
        '''execute keyframes, note this function is blocking call,
        e.g. return until keyframes are executed
        '''
        self.keyframes = keyframes
        return True

    def get_transform(self, name):
        '''get transform with given name'''
        # Compute FK from current sensed joints (radians), same logic as forward_kinematics.py [3](https://fmi100-my.sharepoint.com/personal/isak_schulman_fmi_fi/Documents/Microsoft%20Copilot%20Chat-filer/forward_kinematics.py)
        self.forward_kinematics(self.perception.joint)

        if name in self.chains:
            # return transform of the last joint in that chain
            last_joint = self.chains[name][-1]
            T = self.transforms[last_joint]
        elif name in self.transforms:
            T = self.transforms[name]
        else:
            raise KeyError(f"Unknown name '{name}'. Valid chains: {list(self.chains.keys())}, "
                           f"valid joints: {list(self.transforms.keys())}")

        # Convert numpy array
        return [[float(v) for v in row] for row in T]

    def set_transform(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results'''
        self.inverse_kinematics(effector_name, transform)
        return True


if __name__ == '__main__':
    server = SimpleXMLRPCServer(("localhost", 8000))
    agent = ServerAgent()
    server.register_instance(agent)
    print("Server running on http://localhost:8000")
    server.serve_forever()


