using System;
using System.Collections.Generic;
using UnityEngine;

public class ShadowController : MonoBehaviour
{
    public bool activeWrist = false;
    public ArticulationBody wrist;
    public ArticulationBody palm;
    [Serializable]
    public class ShadowJoint
    {
        public ArticulationBody joint;
        public int questIndex;
        public Vector2 limit;
        public Axis axis;
    }
    public List<ShadowJoint> shadowJoints = new List<ShadowJoint>();

    public enum Axis
    {
        X,
        Y,
        Z
    }
    void Start()
    {
        DataEngine.instance.handTracker.OnFrameInput += Input;
    }
    private void OnDestroy()
    {
        DataEngine.instance.handTracker.OnFrameInput -= Input;
    }
    QuestHandData lastInput;
    Quaternion startWristRot;
    Quaternion rotation;
    new bool enabled;
    void Input(QuestHandData input)
    {
        if (lastInput is null)
            lastInput = input;
        if (input.leftHand.rock)
        {
            if (enabled == false && activeWrist)
            {
                startWristRot = input.rightHand.wristRot;
            }
            enabled = true;
        }
        if (input.leftHand.stop)
            enabled = false;
        if (enabled)
        {
            if (activeWrist && wrist is not null && palm is not null)
            {
                rotation = Quaternion.Inverse(startWristRot) * input.rightHand.wristRot;
                float w = rotation.eulerAngles.y;
                float p = rotation.eulerAngles.z;
                w = NormalizationEuler(w);
                p = NormalizationEuler(p);
                wrist.GetUnit().SetJointPosition(-w);
                palm.GetUnit().SetJointPosition(-p);
            }
            foreach (var joint in shadowJoints)
            {
                if (joint.questIndex < 0) continue;
                float value = 0;
                switch (joint.axis)
                {
                    case Axis.X:
                        value = input.rightHand.joints[joint.questIndex].x;
                        break;
                    case Axis.Y:
                        value = input.rightHand.joints[joint.questIndex].y;
                        break;
                    case Axis.Z:
                        value = input.rightHand.joints[joint.questIndex].z;
                        break;
                }
                value = NormalizationEuler(value);
                var lerp = Mathf.InverseLerp(joint.limit.x, joint.limit.y, value);
                lerp = Mathf.Clamp01(lerp);
                joint.joint.GetUnit().SetJointPosition(Mathf.Lerp(joint.joint.xDrive.lowerLimit, joint.joint.xDrive.upperLimit, lerp));
            }

        }
        lastInput = input;
    }

    float NormalizationEuler(float euler)
    {
        while (euler > 180)
        {
            euler -= 360;
        }
        while (euler < -180)
        {
            euler += 360;
        }
        return euler;
    }
}
