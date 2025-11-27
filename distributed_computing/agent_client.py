'''In this file you need to implement remote procedure call (RPC) client

* The agent_server.py has to be implemented first (at least one function is implemented and exported)
* Please implement functions in ClientAgent first, which should request remote call directly
* The PostHandler can be implement in the last step, it provides non-blocking functions, e.g. agent.post.execute_keyframes
 * Hints: [threading](https://docs.python.org/2/library/threading.html) may be needed for monitoring if the task is done
'''

import weakref
import xmlrpc.client
import threading


class PostHandler(object):
    '''the post handler wraps function to be executed in parallel'''
    def __init__(self, obj):
        self.proxy = weakref.proxy(obj)

    def execute_keyframes(self, keyframes):
        '''non-blocking call of ClientAgent.execute_keyframes'''
        thread = threading.Thread(target=self.proxy.execute_keyframes, args=(keyframes,))
        thread.start()
        return thread

    def set_transform(self, effector_name, transform):
        '''non-blocking call of ClientAgent.set_transform'''
        thread = threading.Thread(target=self.proxy.set_transform, args=(effector_name, transform))
        thread.start()



class ClientAgent(object):
    '''ClientAgent request RPC service from remote server
    '''
    # YOUR CODE HERE
    def __init__(self,server_address):
        self.server = xmlrpc.client.ServerProxy(server_address)
        self.post = PostHandler(self)
    
    def get_angle(self, joint_name):
        '''get sensor value of given joint'''
        return self.server.get_angle(joint_name)
    
    def set_angle(self, joint_name, angle):
        '''set target angle of joint for PID controller
        '''
        return self.server.set_angle(joint_name, angle)


    def get_posture(self):
        '''return current posture of robot'''
        return self.server.get_posture()


    def execute_keyframes(self, keyframes):
        '''excute keyframes, note this function is blocking call,
        e.g. return until keyframes are executed
        '''
        # YOUR CODE HERE
        return self.server.execute_keyframes(keyframes)

    def get_transform(self, name):
        '''get transform with given name
        '''
        # YOUR CODE HERE
        return self.server.get_transform(name)


    def set_transform(self, effector_name, transform):
        '''solve the inverse kinematics and control joints use the results
        '''
        # YOUR CODE HERE
        return self.server.set_transform(effector_name, transform)



if __name__ == '__main__':
    print('hi')
    agent = ClientAgent("http://localhost:8000")
    print('ses')
    # TEST CODE HERE
    print(agent.get_angle("LShoulderPitch"))
    print("Angle get success")

    agent.set_angle("LShoulderPitch", 0.5)
    print("Angle set success")

    print(agent.get_posture()) # 
    print("Posture get success")
    
    
    keyframes = [{"joint": "LShoulderPitch", "angle": 0.5}]
    agent.execute_keyframes(keyframes)
    print("Keyframes execution done (?)")
    
    print(agent.get_transform("LShoulderPitch"))

    agent.set_transform("LArm", [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    print("Transform get/set success")


