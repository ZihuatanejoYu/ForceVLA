using System.Collections.Generic;
using System.Net.Sockets;
using System.Net;
using UnityEngine;
using Newtonsoft.Json;
using System.Text;
using System;
using Oculus.Interaction;

public class HandTracker : MonoBehaviour
{
    public Transform head;
    public OVRSkeleton left;
    public OVRSkeleton right;

    public ActiveStateGroup thumbUpLeft;
    public ActiveStateGroup thumbDownLeft;
    public ActiveStateGroup rockLeft;
    public ActiveStateGroup scissorsLeft;
    public ActiveStateGroup paperLeft;

    public ActiveStateGroup thumbUpRight;
    public ActiveStateGroup thumbDownRight;
    public ActiveStateGroup rockRight;
    public ActiveStateGroup scissorsRight;
    public ActiveStateGroup paperRight;

    UdpClient udpClient = new UdpClient();

    public event Action<QuestHandData> OnFrameInput;

    private void FixedUpdate()
    {
        QuestHandData frame = new QuestHandData();

        frame.headWorldPos = head.position;
        frame.headWorldRot = head.rotation;

        frame.headPos = WorldAlign.instance.GetPosition(frame.headWorldPos);
        frame.headRot = WorldAlign.instance.GetRotation(frame.headWorldRot);

        frame.leftHand = CollectFrame(true);
        frame.rightHand = CollectFrame(false);

        OnFrameInput?.Invoke(frame);

        if (IPAddress.TryParse(DataEngine.instance.targetIP, out IPAddress ip))
        {
            byte[] bytes = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(frame, UnityJsonConverter.JsonSerializerSettings));
            IPEndPoint remotePoint = new IPEndPoint(ip, 10002);
            udpClient.Send(bytes, bytes.Length, remotePoint);
        }
    }

    OneHandData CollectFrame(bool isLeft)
    {
        OVRSkeleton skeleton = isLeft ? left : right;
        OneHandData handFrameData = new OneHandData();

        handFrameData.wristWorldPos = skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_WristRoot].Transform.position;
        handFrameData.wristWorldRot = skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_WristRoot].Transform.rotation;

        handFrameData.wristPos = WorldAlign.instance.GetPosition(handFrameData.wristWorldPos);
        handFrameData.wristRot = WorldAlign.instance.GetRotation(handFrameData.wristWorldRot);

        if (isLeft)
        {
            //handFrameData.wristWorldPos = DataEngine.instance.leftHand.position;
            //handFrameData.wristWorldRot = DataEngine.instance.leftHand.rotation;
            handFrameData.thumbUp = thumbUpLeft.Active;
            handFrameData.thumbDown = thumbDownLeft.Active;
            handFrameData.rock = rockLeft.Active;
            handFrameData.scissors = scissorsLeft.Active;
            handFrameData.stop = paperLeft.Active;
        }
        else
        {
            //handFrameData.wristWorldPos = DataEngine.instance.rightHand.position;
            //handFrameData.wristWorldRot = DataEngine.instance.rightHand.rotation;
            handFrameData.thumbUp = thumbUpRight.Active;
            handFrameData.thumbDown = thumbDownRight.Active;
            handFrameData.rock = rockRight.Active;
            handFrameData.scissors = scissorsRight.Active;
            handFrameData.stop = paperRight.Active;
        }

        handFrameData.joints.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Thumb0].Transform.localEulerAngles);//0
        handFrameData.joints.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Thumb1].Transform.localEulerAngles);//1
        handFrameData.joints.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Thumb2].Transform.localEulerAngles);//2
        handFrameData.joints.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Thumb3].Transform.localEulerAngles);//3

        handFrameData.joints.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Index1].Transform.localEulerAngles);//4
        handFrameData.joints.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Index2].Transform.localEulerAngles);//5
        handFrameData.joints.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Index3].Transform.localEulerAngles);//6

        handFrameData.joints.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Middle1].Transform.localEulerAngles);//7
        handFrameData.joints.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Middle2].Transform.localEulerAngles);//8
        handFrameData.joints.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Middle3].Transform.localEulerAngles);//9

        handFrameData.joints.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Ring1].Transform.localEulerAngles);//10
        handFrameData.joints.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Ring2].Transform.localEulerAngles);//11
        handFrameData.joints.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Ring3].Transform.localEulerAngles);//12

        handFrameData.joints.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Pinky0].Transform.localEulerAngles);//13
        handFrameData.joints.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Pinky1].Transform.localEulerAngles);//14
        handFrameData.joints.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Pinky2].Transform.localEulerAngles);//15
        handFrameData.joints.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Pinky3].Transform.localEulerAngles);//16

        handFrameData.tips.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_ThumbTip].Transform.position);
        handFrameData.tips.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_IndexTip].Transform.position);
        handFrameData.tips.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_MiddleTip].Transform.position);
        handFrameData.tips.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_RingTip].Transform.position);
        handFrameData.tips.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_PinkyTip].Transform.position);

        handFrameData.tipsInWrist.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_WristRoot].Transform.InverseTransformPoint(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_ThumbTip].Transform.position));
        handFrameData.tipsInWrist.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_WristRoot].Transform.InverseTransformPoint(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_IndexTip].Transform.position));
        handFrameData.tipsInWrist.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_WristRoot].Transform.InverseTransformPoint(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_MiddleTip].Transform.position));
        handFrameData.tipsInWrist.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_WristRoot].Transform.InverseTransformPoint(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_RingTip].Transform.position));
        handFrameData.tipsInWrist.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_WristRoot].Transform.InverseTransformPoint(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_PinkyTip].Transform.position));

        handFrameData.joint2InWrist.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_WristRoot].Transform.InverseTransformPoint(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Thumb2].Transform.position));
        handFrameData.joint2InWrist.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_WristRoot].Transform.InverseTransformPoint(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Index2].Transform.position));
        handFrameData.joint2InWrist.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_WristRoot].Transform.InverseTransformPoint(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Middle2].Transform.position));
        handFrameData.joint2InWrist.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_WristRoot].Transform.InverseTransformPoint(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Ring2].Transform.position));
        handFrameData.joint2InWrist.Add(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_WristRoot].Transform.InverseTransformPoint(skeleton.Bones[(int)OVRSkeleton.BoneId.Hand_Pinky2].Transform.position));

        return handFrameData;
    }
}
