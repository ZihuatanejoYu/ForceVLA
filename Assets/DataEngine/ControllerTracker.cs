using Newtonsoft.Json;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class ControllerTracker : MonoBehaviour
{
    public Transform head;

    UdpClient udpClient = new UdpClient();

    public event Action<QuestInputData> OnFrameInput;

    private void FixedUpdate()
    {
        QuestInputData frame = new QuestInputData();

        frame.headWorldPos = head.position;
        frame.headWorldRot = head.rotation;

        frame.headPos = WorldAlign.instance.GetPosition(frame.headWorldPos);
        frame.headRot = WorldAlign.instance.GetRotation(frame.headWorldRot);

        WriteInputData(frame);

        OnFrameInput?.Invoke(frame);

        if (IPAddress.TryParse(DataEngine.instance.targetIP, out IPAddress ip))
        {
            byte[] bytes = Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(frame, UnityJsonConverter.JsonSerializerSettings));
            IPEndPoint remotePoint = new IPEndPoint(ip, 10001);
            udpClient.Send(bytes, bytes.Length, remotePoint);
        }
    }

    void WriteInputData(QuestInputData frame)
    {
        frame.A = OVRInput.Get(OVRInput.RawButton.A);
        frame.B = OVRInput.Get(OVRInput.RawButton.B);
        frame.X = OVRInput.Get(OVRInput.RawButton.X);
        frame.Y = OVRInput.Get(OVRInput.RawButton.Y);

        frame.leftThumb = OVRInput.Get(OVRInput.RawButton.LThumbstick);
        frame.rightThumb = OVRInput.Get(OVRInput.RawButton.RThumbstick);

        frame.leftIndex = OVRInput.Get(OVRInput.RawAxis1D.LIndexTrigger);
        frame.rightIndex = OVRInput.Get(OVRInput.RawAxis1D.RIndexTrigger);

        frame.leftHand = OVRInput.Get(OVRInput.RawAxis1D.LHandTrigger);
        frame.rightHand = OVRInput.Get(OVRInput.RawAxis1D.RHandTrigger);

        frame.leftStick = OVRInput.Get(OVRInput.RawAxis2D.LThumbstick);
        frame.rightStick = OVRInput.Get(OVRInput.RawAxis2D.RThumbstick);

        frame.leftWorldPos = DataEngine.instance.leftHand.position;
        frame.rightWorldPos = DataEngine.instance.rightHand.position;

        frame.leftPos = WorldAlign.instance.GetPosition(frame.leftWorldPos);
        frame.rightPos = WorldAlign.instance.GetPosition(frame.rightWorldPos);

        frame.leftWorldRot = DataEngine.instance.leftHand.rotation;
        frame.rightWorldRot = DataEngine.instance.rightHand.rotation;

        frame.leftRot = WorldAlign.instance.GetRotation(frame.leftWorldRot);
        frame.rightRot = WorldAlign.instance.GetRotation(frame.rightWorldRot);
    }
}
