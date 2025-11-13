'''In this exercise you need to use the learned classifier to recognize current posture of robot

* Tasks:
    1. load learned classifier in `PostureRecognitionAgent.__init__`
    2. recognize current posture in `PostureRecognitionAgent.recognize_posture`

* Hints:
    Let the robot execute different keyframes, and recognize these postures.

'''


from angle_interpolation import AngleInterpolationAgent
from keyframes import hello
import pickle
import os


class PostureRecognitionAgent(AngleInterpolationAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(PostureRecognitionAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.posture = 'unknown'
        
        ROBOT_POSE_CLF = os.path.join('joint_control', 'robot_pose.pkl')
        self.posture_classifier = pickle.load(open(ROBOT_POSE_CLF, "rb"))  # LOAD YOUR CLASSIFIER

    def think(self, perception):
        self.posture = self.recognize_posture(perception)
        return super(PostureRecognitionAgent, self).think(perception)

    def recognize_posture(self, perception):
        # Getting all features from perception
        features = [
            perception.joint['LHipYawPitch'],
            perception.joint['LHipRoll'],
            perception.joint['LHipPitch'],
            perception.joint['LKneePitch'],
            perception.joint['RHipYawPitch'],
            perception.joint['RHipRoll'],
            perception.joint['RHipPitch'],
            perception.joint['RKneePitch'],
            perception.imu[0],
            perception.imu[1]
        ]
        # Reshaping ?
        features = [features]  # shape (1, 10)

        # Posture index prediction
        posture = self.posture_classifier.predict(features)[0]

        # print(posture)

        return posture


if __name__ == '__main__':
    agent = PostureRecognitionAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
