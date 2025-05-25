using System;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

public class ShadowControllerWithIK : MonoBehaviour
{
    [Serializable]
    public class Finger
    {
        public BioIK.BioIK bioIK;
        public Transform tip;
        public Transform joint2;
    }
    public Vector3 handOffset = Vector3.zero;
    public Vector3 handScale = Vector3.one;
    public bool activeWrist = false;
    public ArticulationBody wrist;
    public ArticulationBody palm;
    public List<Finger> fingers = new List<Finger>();

    void Start()
    {
        DataEngine.instance.handTracker.OnFrameInput += Input;
    }
    private void OnDestroy()
    {
        DataEngine.instance.handTracker.OnFrameInput -= Input;
    }
    private void FixedUpdate()
    {
        foreach (var item in fingers)
        {
            item.bioIK.FixedUpdate1();
            foreach (var value in item.bioIK.targets)
            {
                value.Key.GetComponent<ArticulationBody>().GetUnit().SetJointPosition(value.Value);
            }
        }
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
            for (int i = 0; i < 5; i++)
            {
                if (fingers[i].tip is not null)
                    fingers[i].tip.localPosition = new Vector3(-input.rightHand.tipsInWrist[i].y * handScale.x, input.rightHand.tipsInWrist[i].x * handScale.y, input.rightHand.tipsInWrist[i].z * handScale.z) + handOffset;
                if(fingers[i].joint2 is not null)
                    fingers[i].joint2.localPosition = new Vector3(-input.rightHand.joint2InWrist[i].y * handScale.x, input.rightHand.joint2InWrist[i].x * handScale.y, input.rightHand.joint2InWrist[i].z * handScale.z) + handOffset;
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


//#if UNITY_EDITOR
//[CustomEditor(typeof(ShadowControllerWithIK), true)]
//public class ShadowControllerWithIKEditor : Editor
//{
//    public override void OnInspectorGUI()
//    {
//        base.OnInspectorGUI();
//        if (GUILayout.Button("Add IK"))
//        {
//            AddIK();
//        }
//    }
//    private void AddIK()
//    {
//        ShadowControllerWithIK shadowController = (ShadowControllerWithIK)target;
//        foreach (var trans in shadowController.roots)
//        {
//            List<ArticulationBody> Joints = trans.GetComponentsInChildren<ArticulationBody>().ToList();

//            Transform iKFollow = new GameObject("iKFollowPoint").transform;
//            iKFollow.SetParent(Joints.Last().transform);
//            iKFollow.localPosition = Vector3.zero;
//            iKFollow.localRotation = Quaternion.identity;

//            Transform iKTarget = new GameObject("iKTargetPoint").transform;
//            iKTarget.parent = trans;

//            iKTarget.position = iKFollow.position;
//            iKTarget.rotation = iKFollow.rotation;

//            BioIK.BioIK bioIK = Joints.First().gameObject.AddComponent<BioIK.BioIK>();
//            bioIK.isArticulations = true;
//            bioIK.SetGenerations(10);
//            bioIK.SetPopulationSize(50);
//            bioIK.SetElites(1);
//            bioIK.Smoothing = 0;
//            bioIK.Refresh();
//            foreach (var item in Joints)
//            {
//                BioIK.BioJoint joint = bioIK.FindSegment(item.transform).AddJoint();
//                joint.SetAnchor(item.anchorPosition);
//                joint.SetOrientation(item.anchorRotation.eulerAngles);
//                joint.SetDefaultFrame(item.transform.localPosition, item.transform.localRotation);
//                switch (item.jointType)
//                {
//                    case ArticulationJointType.RevoluteJoint:
//                        joint.JointType = BioIK.JointType.Rotational;
//                        joint.X.Enabled = true;
//                        switch (item.twistLock)
//                        {
//                            case ArticulationDofLock.FreeMotion:
//                                joint.X.Constrained = false;
//                                break;
//                            case ArticulationDofLock.LimitedMotion:
//                                joint.X.Constrained = true;
//                                joint.X.SetUpperLimit(item.xDrive.upperLimit);
//                                joint.X.SetLowerLimit(item.xDrive.lowerLimit);
//                                joint.X.SetTargetValue(item.xDrive.target);
//                                break;
//                        }
//                        joint.Y.Enabled = false;
//                        joint.Z.Enabled = false;
//                        break;
//                    case ArticulationJointType.PrismaticJoint:
//                        joint.JointType = BioIK.JointType.Translational;
//                        switch (item.linearLockX)
//                        {
//                            case ArticulationDofLock.LockedMotion:
//                                joint.X.Enabled = false;
//                                break;
//                            case ArticulationDofLock.FreeMotion:
//                                joint.X.Enabled = true;
//                                joint.X.Constrained = false;
//                                break;
//                            case ArticulationDofLock.LimitedMotion:
//                                joint.X.Enabled = true;
//                                joint.X.Constrained = true;
//                                joint.X.SetUpperLimit(item.xDrive.upperLimit);
//                                joint.X.SetLowerLimit(item.xDrive.lowerLimit);
//                                joint.X.SetTargetValue(item.xDrive.target);
//                                break;
//                        }
//                        switch (item.linearLockY)
//                        {
//                            case ArticulationDofLock.LockedMotion:
//                                joint.Y.Enabled = false;
//                                break;
//                            case ArticulationDofLock.FreeMotion:
//                                joint.Y.Enabled = true;
//                                joint.Y.Constrained = false;
//                                break;
//                            case ArticulationDofLock.LimitedMotion:
//                                joint.Y.Enabled = true;
//                                joint.Y.Constrained = true;
//                                joint.Y.SetUpperLimit(item.yDrive.upperLimit);
//                                joint.Y.SetLowerLimit(item.yDrive.lowerLimit);
//                                joint.Y.SetTargetValue(item.yDrive.target);
//                                break;
//                        }
//                        switch (item.linearLockZ)
//                        {
//                            case ArticulationDofLock.LockedMotion:
//                                joint.Z.Enabled = false;
//                                break;
//                            case ArticulationDofLock.FreeMotion:
//                                joint.Z.Enabled = true;
//                                joint.Z.Constrained = false;
//                                break;
//                            case ArticulationDofLock.LimitedMotion:
//                                joint.Z.Enabled = true;
//                                joint.Z.Constrained = true;
//                                joint.Z.SetUpperLimit(item.zDrive.upperLimit);
//                                joint.Z.SetLowerLimit(item.zDrive.lowerLimit);
//                                joint.Z.SetTargetValue(item.zDrive.target);
//                                break;
//                        }
//                        break;
//                    case ArticulationJointType.SphericalJoint:
//                        joint.JointType = BioIK.JointType.Rotational;
//                        switch (item.twistLock)
//                        {
//                            case ArticulationDofLock.LockedMotion:
//                                joint.X.Enabled = false;
//                                break;
//                            case ArticulationDofLock.FreeMotion:
//                                joint.X.Enabled = true;
//                                joint.X.Constrained = false;
//                                break;
//                            case ArticulationDofLock.LimitedMotion:
//                                joint.X.Enabled = true;
//                                joint.X.Constrained = true;
//                                joint.X.SetUpperLimit(item.xDrive.upperLimit);
//                                joint.X.SetLowerLimit(item.xDrive.lowerLimit);
//                                joint.X.SetTargetValue(item.xDrive.target);
//                                break;
//                        }
//                        switch (item.swingYLock)
//                        {
//                            case ArticulationDofLock.LockedMotion:
//                                joint.Y.Enabled = false;
//                                break;
//                            case ArticulationDofLock.FreeMotion:
//                                joint.Y.Enabled = true;
//                                joint.Y.Constrained = false;
//                                break;
//                            case ArticulationDofLock.LimitedMotion:
//                                joint.Y.Enabled = true;
//                                joint.Y.Constrained = true;
//                                joint.Y.SetUpperLimit(item.yDrive.upperLimit);
//                                joint.Y.SetLowerLimit(item.yDrive.lowerLimit);
//                                joint.Y.SetTargetValue(item.yDrive.target);
//                                break;
//                        }
//                        switch (item.swingZLock)
//                        {
//                            case ArticulationDofLock.LockedMotion:
//                                joint.Z.Enabled = false;
//                                break;
//                            case ArticulationDofLock.FreeMotion:
//                                joint.Z.Enabled = true;
//                                joint.Z.Constrained = false;
//                                break;
//                            case ArticulationDofLock.LimitedMotion:
//                                joint.Z.Enabled = true;
//                                joint.Z.Constrained = true;
//                                joint.Z.SetUpperLimit(item.zDrive.upperLimit);
//                                joint.Z.SetLowerLimit(item.zDrive.lowerLimit);
//                                joint.Z.SetTargetValue(item.zDrive.target);
//                                break;
//                        }
//                        break;
//                }
//            }
//            if (iKFollow != null)
//            {
//                BioIK.BioSegment segment = bioIK.FindSegment(iKFollow);
//                segment.Objectives = new BioIK.BioObjective[] { };
//                BioIK.BioObjective positionObjective = segment.AddObjective(BioIK.ObjectiveType.Position);
//                ((BioIK.Position)positionObjective).SetTargetTransform(iKTarget);
//            }
//            bioIK.Refresh();
//        }
//    }
//}
//# endif
