'''In this exercise you need to implement an angle interploation function which makes NAO executes keyframe motion

* Tasks:
    1. complete the code in `AngleInterpolationAgent.angle_interpolation`,
       you are free to use splines interploation or Bezier interploation,
       but the keyframes provided are for Bezier curves, you can simply ignore some data for splines interploation,
       please refer data format below for details.
    2. try different keyframes from `keyframes` folder

* Keyframe data format:
    keyframe := (names, times, keys)
    names := [str, ...]  # list of joint names
    times := [[float, float, ...], [float, float, ...], ...]
    # times is a matrix of floats: Each line corresponding to a joint, and column element to a key.
    keys := [[float, [int, float, float], [int, float, float]], ...]
    # keys is a list of angles in radians or an array of arrays each containing [float angle, Handle1, Handle2],
    # where Handle is [int InterpolationType, float dTime, float dAngle] describing the handle offsets relative
    # to the angle and time of the point. The first Bezier param describes the handle that controls the curve
    # preceding the point, the second describes the curve following the point.
'''


from pid import PIDAgent
from keyframes import wipe_forehead
from keyframes import hello
from spark_agent import JOINT_CMD_NAMES


class AngleInterpolationAgent(PIDAgent):
    def __init__(self, simspark_ip='localhost',
                 simspark_port=3100,
                 teamname='DAInamite',
                 player_id=0,
                 sync_mode=True):
        super(AngleInterpolationAgent, self).__init__(simspark_ip, simspark_port, teamname, player_id, sync_mode)
        self.keyframes = ([], [], [])

    def think(self, perception):
        target_joints = self.angle_interpolation(self.keyframes, perception)
        target_joints['RHipYawPitch'] = target_joints['LHipYawPitch'] # copy missing joint in keyframes
        self.target_joints.update(target_joints)
        return super(AngleInterpolationAgent, self).think(perception)

    def angle_interpolation(self, keyframes, perception):
        target_joints = {}
        
        # Accessing keyframes
        names, times, keys = keyframes

        # Current time for the movement in question
        if len(times) == 0:
            current_time = perception.time
        else:

            ntime = len(times[0])-1
            current_time = perception.time % times[0][ntime]  

        # Go through each keyframe and time for each joint in the action
        for i, joint_name in enumerate(names):
            t_seq = times[i]
            # Secure way of handilng keyframes
            k_seq = [k[0] if isinstance(k, list) else k for k in keys[i]]

            # Find the two keyframes closest to current_time
            for j in range(len(t_seq) - 1):
                if t_seq[j] <= current_time <= t_seq[j + 1]:
                    ratio = (current_time - t_seq[j]) / (t_seq[j + 1] - t_seq[j])
                    angle = k_seq[j] + ratio * (k_seq[j + 1] - k_seq[j])
                    target_joints[joint_name] = angle
                    break
            else:
                target_joints[joint_name] = k_seq[-1]

        # Fill missing joints with current positions to prevent error
        for joint in JOINT_CMD_NAMES.keys():
            if joint not in target_joints:
                target_joints[joint] = perception.joint.get(joint, 0.0)

        return target_joints

if __name__ == '__main__':
    agent = AngleInterpolationAgent()
    agent.keyframes = hello()  # CHANGE DIFFERENT KEYFRAMES
    agent.run()
