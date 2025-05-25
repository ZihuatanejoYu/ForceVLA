using Newtonsoft.Json;
using Newtonsoft.Json.Bson;
using System;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

public class ImageSync : MonoBehaviour
{
    public int port;
    Thread th;
    UdpClient server;
    Texture2D texture;

    JsonSerializer serializer = new JsonSerializer();
    private void Start()
    {
        th = new Thread(Server);
        th.Start();
        texture = new Texture2D(1, 1);
        GetComponent<Renderer>().material.mainTexture = texture;
    }
    void Server()
    {
        server = new UdpClient(port);
        IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);
        while (true)
        {
            try
            {
                ReceiveImage(server, remoteEndPoint);
            }
            catch (Exception e)
            {
                Debug.LogError($"SocketException: {e.Message}");
            }
        }
    }

    List<byte> bytes = new List<byte>();
    void ReceiveImage(UdpClient serverImage, IPEndPoint remoteEndPoint)
    {
        byte[] lengthBytes = serverImage.Receive(ref remoteEndPoint);
        if (lengthBytes.Length != 4)
            throw new Exception();
        uint lengthInt = BitConverter.ToUInt32(lengthBytes, 0);
        byte[] chunkBytes = serverImage.Receive(ref remoteEndPoint);
        if (chunkBytes.Length != 4)
            throw new Exception();
        uint lengthChunk = BitConverter.ToUInt32(chunkBytes, 0);
        bytes.Clear();
        int count = Mathf.CeilToInt(lengthInt / (float)lengthChunk);
        for (int index = 0; index < count; index++)
        {
            byte[] buffer = serverImage.Receive(ref remoteEndPoint);
            bytes.AddRange(buffer);
        }
        using (MemoryStream ms = new MemoryStream(bytes.ToArray()))
        {
            using (BsonReader reader = new BsonReader(ms))
            {
                ImageData imageData = serializer.Deserialize<ImageData>(reader);
                UnityMainThreadDispatcher.Instance().Enqueue(() => UpdateImage(imageData));
            }
        }
    }


    void UpdateImage(ImageData frameData)
    {
        if (frameData == null) return;
        if (frameData.updatePose)
        {
            if(frameData.inHeadSpace)
                transform.SetParent(DataEngine.instance.head);
            else
                transform.SetParent(WorldAlign.instance.transform);

            transform.localPosition = new Vector3(frameData.position[0], frameData.position[1], frameData.position[2]);
            if (frameData.rotation != null)
                transform.localRotation = new Quaternion(frameData.rotation[0], frameData.rotation[1], frameData.rotation[2], frameData.rotation[3]);
            else if (frameData.eulerAngle != null)
                transform.localEulerAngles = new Vector3(frameData.eulerAngle[0], frameData.eulerAngle[1], frameData.eulerAngle[2]);
            transform.localScale = new Vector3(frameData.scale[0], frameData.scale[1], frameData.scale[2]);
        }
        texture.LoadImage(frameData.image, true);
    }

    private void OnDestroy()
    {
        server?.Close();
        server?.Dispose();
        th.Abort();
    }
}
