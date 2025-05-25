using System.Collections.Generic;
using UnityEngine;

public class OneHandData
{
    public Vector3 wristPos;
    public Quaternion wristRot;

    public Vector3 wristWorldPos;
    public Quaternion wristWorldRot;

    public List<Vector3> joints = new List<Vector3>();
    public List<Vector3> tips = new List<Vector3>();

    public List<Vector3> tipsInWrist = new List<Vector3>();
    public List<Vector3> joint2InWrist = new List<Vector3>();

    public bool thumbUp;
    public bool thumbDown;
    public bool rock;
    public bool scissors;
    public bool stop;
}

public class QuestHandData
{
    public Vector3 headPos;
    public Quaternion headRot;

    public Vector3 headWorldPos;
    public Quaternion headWorldRot;

    public OneHandData leftHand;
    public OneHandData rightHand;
}
