using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

[Serializable]
public class EmgPacket
{
    public double timestamp;   // seconds since epoch
    public string input;       // e.g. "emg"
    public float value;        // e.g. 0.123
}

public class EMGReceiver : MonoBehaviour
{
    // Must match the port used by UnitySender in Python
    public int listenPort = 6000;

    // If true, print each received packet to the Console
    public bool debugLog = true;

    // Stores the latest activation for each input name (e.g. "emg")
    private Dictionary<string, float> _latestValues = new Dictionary<string, float>();

    // TCP listener/client and reader
    private TcpListener _listener;
    private TcpClient _client;
    private StreamReader _reader;

    // Flag to indicate that a new client connected and we should start reading
    private bool _shouldStartReadLoop = false;

    void Start()
    {
        _listener = new TcpListener(IPAddress.Loopback, listenPort);
        _listener.Start();
        if (debugLog)
        {
            Debug.Log($"EMGReceiver: Listening on port {listenPort}");
        }

        // Asynchronously accept a Python client
        _listener.BeginAcceptTcpClient(OnClientConnected, null);
    }

    private void OnClientConnected(IAsyncResult ar)
    {
        try
        {
            _client = _listener.EndAcceptTcpClient(ar);
            if (debugLog)
            {
                Debug.Log("EMGReceiver: Python connected.");
            }

            // Wrap the network stream in a StreamReader for ReadLine()
            _reader = new StreamReader(_client.GetStream(), Encoding.UTF8);

            // Instead of starting the coroutine here (which is not on the main thread),
            // we set a flag and let Update() start it on the main thread.
            _shouldStartReadLoop = true;

            // If you want to accept further reconnections, call BeginAcceptTcpClient again:
            // _listener.BeginAcceptTcpClient(OnClientConnected, null);
        }
        catch (Exception e)
        {
            Debug.LogError("EMGReceiver: Error in OnClientConnected: " + e);
        }
    }

    private IEnumerator ReadLoop()
    {
        while (_client != null && _client.Connected)
        {
            string line = null;
            try
            {
                // Only read when data is available
                if (_reader.Peek() >= 0)
                {
                    line = _reader.ReadLine();
                }
            }
            catch (Exception e)
            {
                Debug.LogError("EMGReceiver: Socket read error: " + e);
                break;
            }

            if (!string.IsNullOrEmpty(line))
            {
                EmgPacket packet = JsonUtility.FromJson<EmgPacket>(line);
                if (packet != null)
                {
                    _latestValues[packet.input] = packet.value;
                    if (debugLog)
                    {
                        Debug.Log($"EMGReceiver: {packet.input} → {packet.value:0.000}");
                    }
                }
                else if (debugLog)
                {
                    Debug.LogWarning($"EMGReceiver: Could not parse JSON: {line}");
                }
            }

            // Yield one frame so Unity’s main thread is not blocked
            yield return null;
        }
    }

    void Update()
    {
        // If the flag was set by OnClientConnected, start the coroutine now (on main thread)
        if (_shouldStartReadLoop)
        {
            _shouldStartReadLoop = false;
            StartCoroutine(ReadLoop());
        }

        // Example: map "emg" activation to a global shader float (0..1)
        if (_latestValues.TryGetValue("emg", out float emgVal))
        {
            float clamped = Mathf.Clamp01(emgVal);
            Shader.SetGlobalFloat("_EmgActivation", clamped);
        }
    }

    private void OnApplicationQuit()
    {
        if (_reader != null) _reader.Close();
        if (_client != null) _client.Close();
        if (_listener != null) _listener.Stop();
    }
}
