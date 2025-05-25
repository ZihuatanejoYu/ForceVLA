using System.Collections.Generic;
using UnityEngine;

public class ArticulationsJointData
{
    public List<float> jointPositions = new List<float>();
}
public class ArticulationsTargetData
{
    public bool isRelative;
    public Vector3 position;
    public Quaternion rotation;
}
