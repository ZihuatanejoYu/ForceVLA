using System;
using System.Collections.Generic;
using UnityEngine;

public enum LeftOrRight
{
    Left,
    Right
}

public class GripperController : MonoBehaviour
{
    [Serializable]
    public class GripperJoint
    {
        public ArticulationBody joint;
        public Vector2 limit;
    }

    public List<GripperJoint> gripperJoints = new List<GripperJoint>();


    public LeftOrRight hand;
    void Start()
    {
        DataEngine.instance.controllerTracker.OnFrameInput += ControllerInput;
        DataEngine.instance.handTracker.OnFrameInput += HandInput;
    }
    private void OnDestroy()
    {
        DataEngine.instance.controllerTracker.OnFrameInput -= ControllerInput;
        DataEngine.instance.handTracker.OnFrameInput -= HandInput;
    }
    void ControllerInput(QuestInputData input)
    {
        float value;
        switch (hand)
        {
            case LeftOrRight.Left:
                value = input.leftIndex;
                break;
            default:
                value = input.rightIndex;
                break;
        }
        foreach (var joint in gripperJoints)
        {
            joint.joint.GetUnit().SetJointPosition(Mathf.Lerp(joint.limit.x, joint.limit.y, value));
        }
    }

    void HandInput(QuestHandData input)
    {
        bool run;
        switch (hand)
        {
            case LeftOrRight.Left:
                run = input.rightHand.rock;
                break;
            default:
                run = input.leftHand.rock;
                break;
        }
        if (run)
        {
            float value;
            switch (hand)
            {
                case LeftOrRight.Left:
                    value = Vector3.Distance(input.leftHand.tips[0], input.leftHand.tips[1]);
                    break;
                default:
                    value = Vector3.Distance(input.rightHand.tips[0], input.rightHand.tips[1]);
                    break;
            }
            value = Mathf.InverseLerp(0.01f, 0.08f, value);
            value = Mathf.Clamp01(value);
            foreach (var joint in gripperJoints)
            {
                joint.joint.GetUnit().SetJointPosition(Mathf.Lerp(joint.limit.x, joint.limit.y, value));
            }
        }

    }
}
