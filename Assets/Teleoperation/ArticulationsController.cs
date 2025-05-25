using System.Collections.Generic;
using System.Linq;
using System.Net.Sockets;
using System.Net;
using UnityEngine;
using System.Text;
using Newtonsoft.Json;
using System.Threading;
using UnityEditor;
using System;


public class ArticulationsController : MonoBehaviour
{
    public bool BIOIK => bioIK is not null && bioIK.enabled;
    public int port;
    public LeftOrRight hand;
    public bool sendOrReceive;
    public List<ArticulationsController> childs;
    UdpClient udpClient;
    Thread thread;
    private void Start()
    {
        udpClient = new UdpClient(port);
        if (BIOIK)
        {
            DataEngine.instance.controllerTracker.OnFrameInput += ControllerInput;
            DataEngine.instance.handTracker.OnFrameInput += HandInput;
        }
        else
        {
            thread = new Thread(ReceiveThread);
            thread.Start();
        }

    }
    private void OnDestroy()
    {
        udpClient.Close();
        if (BIOIK)
        {
            DataEngine.instance.controllerTracker.OnFrameInput -= ControllerInput;
            DataEngine.instance.handTracker.OnFrameInput -= HandInput;
        }
        else
            thread?.Abort();
    }

    QuestInputData lastInputData;
    void ControllerInput(QuestInputData inputData)
    {
        switch(hand)
        {
            case LeftOrRight.Right:
                if (inputData.rightHand > 0.5f)
                {
                    iKTarget.position += inputData.rightWorldPos - lastInputData.rightWorldPos;
                    Quaternion rotationDifference = inputData.rightWorldRot * Quaternion.Inverse(lastInputData.rightWorldRot);
                    iKTarget.rotation = rotationDifference * iKTarget.rotation;
                }
                break;
            case LeftOrRight.Left:
                if (inputData.leftHand > 0.5f)
                {
                    iKTarget.position += inputData.leftWorldPos - lastInputData.leftWorldPos;
                    Quaternion rotationDifference = inputData.leftWorldRot * Quaternion.Inverse(lastInputData.leftWorldRot);
                    iKTarget.rotation = rotationDifference * iKTarget.rotation;
                }
                break;

        }
        lastInputData = inputData;
    }

    QuestHandData lastHandData;
    new bool enabled;
    void HandInput(QuestHandData inputData)
    {
        if (inputData.leftHand.rock)
            enabled = true;
        if (inputData.leftHand.stop)
            enabled = false;
        if (enabled)
        {
            iKTarget.position += inputData.rightHand.wristWorldPos - lastHandData.rightHand.wristWorldPos;
            Quaternion rotationDifference = inputData.rightHand.wristWorldRot * Quaternion.Inverse(lastHandData.rightHand.wristWorldRot);
            iKTarget.rotation = rotationDifference * iKTarget.rotation;
        }
        lastHandData = inputData;
    }
    private void FixedUpdate()
    {
        if (BIOIK)
        {
            bioIK.FixedUpdate1();
            foreach (var item in bioIK.targets)
            {
                iKCopy[item.Key].GetUnit().SetJointPosition(item.Value);
            }
        }
        if (sendOrReceive)
        {
            if (IPAddress.TryParse(DataEngine.instance.targetIP, out IPAddress ip))
            {
                ArticulationsJointData frame = new ArticulationsJointData();
                frame.jointPositions = MoveableJoints.Select((s) => s.GetUnit().GetJointPosition()).ToList();
                byte[] bytes = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(frame, UnityJsonConverter.JsonSerializerSettings));
                IPEndPoint remotePoint = new IPEndPoint(ip, port);
                udpClient.Send(bytes, bytes.Length, remotePoint);
            }
        }
    }

    void ReceiveThread()
    {
        IPEndPoint remotePoint = new IPEndPoint(IPAddress.Any, 0);
        while (true)
        {
            try
            {
                byte[] bytes = udpClient.Receive(ref remotePoint);
                ArticulationsJointData data = JsonConvert.DeserializeObject<ArticulationsJointData>(Encoding.UTF8.GetString(bytes));
                UnityMainThreadDispatcher.Instance().Enqueue(() => UpdateArticulations(data));
            }
            catch (Exception e)
            {
                Debug.LogError($"SocketException: {e.Message}");
            }
        }
    }

    void UpdateArticulations(ArticulationsJointData data)
    {
        if (data == null) return;
        for (int i = 0; i < data.jointPositions.Count; i++)
        {
            if (i < moveableJoints.Count)
                MoveableJoints[i].GetUnit().SetJointPositionDirectly(data.jointPositions[i]);
        }
    }
    void UpdateArticulations(ArticulationsTargetData data)
    {
        if (data == null) return;
        if (data.isRelative)
            SetIKTarget(iKTarget.position + data.position, iKTarget.rotation * data.rotation);
        else
            SetIKTarget(data.position, data.rotation);
    }

    public Transform iKFollow;
    public Transform iKTarget;

    BioIK.BioIK bioIK = null;

    private Dictionary<Transform, ArticulationBody> iKCopy = new();
    public void InitBioIK()
    {
        if (Joints.Count == 0) return;

        iKFollow = new GameObject("iKFollowPoint").transform;
        iKFollow.SetParent(Joints.Last().transform);
        iKFollow.localPosition = Vector3.zero;
        iKFollow.localRotation = Quaternion.identity;

        iKTarget = new GameObject("iKTargetPoint").transform;
        iKTarget.parent = transform;
        ResetIKTarget();

        bioIK = Joints.First().gameObject.AddComponent<BioIK.BioIK>();
        bioIK.isArticulations = true;
        bioIK.SetGenerations(10);
        bioIK.SetPopulationSize(50);
        bioIK.SetElites(1);
        bioIK.Smoothing = 0;
        foreach (var item in MoveableJoints)
        {
            BioIK.BioJoint joint = bioIK.FindSegment(item.transform).AddJoint();
            iKCopy[item.transform] = item;
            joint.SetAnchor(item.anchorPosition);
            joint.SetOrientation(item.anchorRotation.eulerAngles);
            joint.SetDefaultFrame(item.transform.localPosition, item.transform.localRotation);
            switch (item.jointType)
            {
                case ArticulationJointType.RevoluteJoint:
                    joint.JointType = BioIK.JointType.Rotational;
                    joint.X.Enabled = true;
                    switch (item.twistLock)
                    {
                        case ArticulationDofLock.FreeMotion:
                            joint.X.Constrained = false;
                            break;
                        case ArticulationDofLock.LimitedMotion:
                            joint.X.Constrained = true;
                            joint.X.SetUpperLimit(item.xDrive.upperLimit);
                            joint.X.SetLowerLimit(item.xDrive.lowerLimit);
                            joint.X.SetTargetValue(item.xDrive.target);
                            break;
                    }
                    joint.Y.Enabled = false;
                    joint.Z.Enabled = false;
                    break;
                case ArticulationJointType.PrismaticJoint:
                    joint.JointType = BioIK.JointType.Translational;
                    switch (item.linearLockX)
                    {
                        case ArticulationDofLock.LockedMotion:
                            joint.X.Enabled = false;
                            break;
                        case ArticulationDofLock.FreeMotion:
                            joint.X.Enabled = true;
                            joint.X.Constrained = false;
                            break;
                        case ArticulationDofLock.LimitedMotion:
                            joint.X.Enabled = true;
                            joint.X.Constrained = true;
                            joint.X.SetUpperLimit(item.xDrive.upperLimit);
                            joint.X.SetLowerLimit(item.xDrive.lowerLimit);
                            joint.X.SetTargetValue(item.xDrive.target);
                            break;
                    }
                    switch (item.linearLockY)
                    {
                        case ArticulationDofLock.LockedMotion:
                            joint.Y.Enabled = false;
                            break;
                        case ArticulationDofLock.FreeMotion:
                            joint.Y.Enabled = true;
                            joint.Y.Constrained = false;
                            break;
                        case ArticulationDofLock.LimitedMotion:
                            joint.Y.Enabled = true;
                            joint.Y.Constrained = true;
                            joint.Y.SetUpperLimit(item.yDrive.upperLimit);
                            joint.Y.SetLowerLimit(item.yDrive.lowerLimit);
                            joint.Y.SetTargetValue(item.yDrive.target);
                            break;
                    }
                    switch (item.linearLockZ)
                    {
                        case ArticulationDofLock.LockedMotion:
                            joint.Z.Enabled = false;
                            break;
                        case ArticulationDofLock.FreeMotion:
                            joint.Z.Enabled = true;
                            joint.Z.Constrained = false;
                            break;
                        case ArticulationDofLock.LimitedMotion:
                            joint.Z.Enabled = true;
                            joint.Z.Constrained = true;
                            joint.Z.SetUpperLimit(item.zDrive.upperLimit);
                            joint.Z.SetLowerLimit(item.zDrive.lowerLimit);
                            joint.Z.SetTargetValue(item.zDrive.target);
                            break;
                    }
                    break;
                case ArticulationJointType.SphericalJoint:
                    joint.JointType = BioIK.JointType.Rotational;
                    switch (item.twistLock)
                    {
                        case ArticulationDofLock.LockedMotion:
                            joint.X.Enabled = false;
                            break;
                        case ArticulationDofLock.FreeMotion:
                            joint.X.Enabled = true;
                            joint.X.Constrained = false;
                            break;
                        case ArticulationDofLock.LimitedMotion:
                            joint.X.Enabled = true;
                            joint.X.Constrained = true;
                            joint.X.SetUpperLimit(item.xDrive.upperLimit);
                            joint.X.SetLowerLimit(item.xDrive.lowerLimit);
                            joint.X.SetTargetValue(item.xDrive.target);
                            break;
                    }
                    switch (item.swingYLock)
                    {
                        case ArticulationDofLock.LockedMotion:
                            joint.Y.Enabled = false;
                            break;
                        case ArticulationDofLock.FreeMotion:
                            joint.Y.Enabled = true;
                            joint.Y.Constrained = false;
                            break;
                        case ArticulationDofLock.LimitedMotion:
                            joint.Y.Enabled = true;
                            joint.Y.Constrained = true;
                            joint.Y.SetUpperLimit(item.yDrive.upperLimit);
                            joint.Y.SetLowerLimit(item.yDrive.lowerLimit);
                            joint.Y.SetTargetValue(item.yDrive.target);
                            break;
                    }
                    switch (item.swingZLock)
                    {
                        case ArticulationDofLock.LockedMotion:
                            joint.Z.Enabled = false;
                            break;
                        case ArticulationDofLock.FreeMotion:
                            joint.Z.Enabled = true;
                            joint.Z.Constrained = false;
                            break;
                        case ArticulationDofLock.LimitedMotion:
                            joint.Z.Enabled = true;
                            joint.Z.Constrained = true;
                            joint.Z.SetUpperLimit(item.zDrive.upperLimit);
                            joint.Z.SetLowerLimit(item.zDrive.lowerLimit);
                            joint.Z.SetTargetValue(item.zDrive.target);
                            break;
                    }
                    break;
            }
        }

        if (iKFollow != null)
        {
            BioIK.BioSegment segment = bioIK.FindSegment(iKFollow);
            segment.Objectives = new BioIK.BioObjective[] { };
            BioIK.BioObjective positionObjective = segment.AddObjective(BioIK.ObjectiveType.Position);
            ((BioIK.Position)positionObjective).SetTargetTransform(iKTarget);
            //if (iKTargetOrientation)
            {
                BioIK.BioObjective orientationObjective = segment.AddObjective(BioIK.ObjectiveType.Orientation);
                ((BioIK.Orientation)orientationObjective).SetTargetTransform(iKTarget);
            }
        }
        bioIK.Refresh();
    }
    public void SetPositionAndRotation(Vector3 pos, Quaternion quat)
    {
        transform.position = pos;
        transform.rotation = quat;
        root.TeleportRoot(transform.position, transform.rotation);
    }
    void DirectlyIK(Vector3 targetPos, Quaternion targetRot)
    {
        Debug.Log("DirectlyIK");
        //iKTarget.DOKill();
        float disPos = Vector3.Distance(iKTarget.position, targetPos);
        float disRot = Quaternion.Angle(iKTarget.rotation, targetRot);
        int lerpCountPos = (int)(disPos / 0.01f) + 1;
        int lerpCountRot = (int)(disRot / 1f) + 1;
        int lerpCount = Mathf.Max(lerpCountPos, lerpCountRot, 5);
        Vector3 startPosition = iKTarget.position;
        Quaternion startQuaternion = iKTarget.rotation;
        //foreach (var item in MoveableJoints)
        //{
        //    BioIK.BioJoint joint = bioIK.FindSegment(item.transform).AddJoint();
        //    joint.SetDefaultFrame(item.transform.localPosition, item.transform.localRotation);
        //}
        for (int i = 0; i < lerpCount; i++)
        {
            iKTarget.position = Vector3.Lerp(startPosition, targetPos, i + 1 / (float)lerpCount);
            iKTarget.rotation = Quaternion.Lerp(startQuaternion, targetRot, i + 1 / (float)lerpCount);
            bioIK.FixedUpdate1();
        }
        foreach (var item in bioIK.targets)
        {
            iKCopy[item.Key].GetUnit().SetJointPositionDirectly(item.Value);
        }
    }
    void SetIKTarget(Vector3 position, Quaternion rotation)
    {
        if (iKFollow != null && iKTarget != null)
        {
            iKTarget.position = position;
            iKTarget.rotation = rotation;
        }
    }
    void ResetIKTarget()
    {
        if (iKFollow != null && iKTarget != null)
        {
            iKTarget.position = iKFollow.position;
            iKTarget.rotation = iKFollow.rotation;
        }
    }
    public ArticulationBody root => Joints.First();
    public bool Immovable
    {
        get { return root.immovable; }
        set { root.immovable = value; }
    }

    List<ArticulationBody> joints;
    List<ArticulationBody> Joints
    {
        get
        {
            if (joints == null)
                InitJoints();
            return joints;
        }
    }
    List<ArticulationBody> moveableJoints;
    List<ArticulationBody> MoveableJoints
    {
        get
        {
            if (moveableJoints == null)
                InitJoints();
            return moveableJoints;
        }
    }
    public void InitJoints()
    {
        joints = new List<ArticulationBody>();
        moveableJoints = new List<ArticulationBody>();
        foreach (var item in GlobalUtility.GetChildComponentFilter<ArticulationsController, ArticulationBody>(this))
        {
            joints.Add(item);
            if (item.jointType != ArticulationJointType.FixedJoint && item.GetUnit().mimicParent == null && !item.isRoot)
                moveableJoints.Add(item);
        }
    }

}
