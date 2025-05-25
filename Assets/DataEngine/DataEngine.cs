using System.Threading.Tasks;
using TMPro;
using UnityEngine;
using UnityEngine.UI;


public class DataEngine : MonoBehaviour
{
    public static DataEngine instance;

    public Transform head;
    public Button keyboardUI;
    public Button handUI;
    public Button controllerUI;
    public Button frankaVR;
    public Button frankaReal;
    public Button flexivVR;
    public Button flexivReal;
    public Button ur10ShadowVR;
    public Button ur10ShadowReal;
    public Button shadowVR;
    public Button shadowReal;
    public Button bisionVR;
    public Button bisionReal;
    public Button shadowVRwithIK;
    public Button imageUI;

    public OVRVirtualKeyboard keyboard;

    public TextMeshProUGUI testLog;
    public HandTracker handTracker;
    public ControllerTracker controllerTracker;

    public ArticulationsController frankaPrefab;
    public ArticulationsController flexivPrefab;
    public ArticulationsController ur10ShadowPrefab;
    public ArticulationsController shadowPrefab;
    public ArticulationsController bisonPrefab;
    public ArticulationsController shadowWithIKPrefab;
    public GameObject imagePrefab;

    public Transform leftHand;
    public Transform rightHand;

    ArticulationsController currentRobot;
    GameObject currentImage;
    public string targetIP => keyboard.TextHandler.Text;

    bool calibrationMode = false;
    bool CalibrationMode
    {
        get
        {
            return calibrationMode;
        }
        set
        {
            if (value == calibrationMode) return;
            calibrationMode = value;
            WorldAlign.instance.SwitchAlign(value);
        }
    }

    private void Awake()
    {
        instance = this;
    }

    bool cd = false;

    async void SwitchMode()
    {
        if (cd) return;
        cd = true;
        CalibrationMode = !CalibrationMode;
        await Task.Delay(500);
        cd = false;
    }
    private void Update()
    {
        if (OVRInput.Get(OVRInput.RawButton.X) && OVRInput.Get(OVRInput.RawButton.A))
        {
            SwitchMode();
        }
        if (Input.GetKeyDown(KeyCode.L))
        {
            LoadBison(bisonPrefab, true, true);
        }


    }
    void Start()
    {
        keyboardUI.onClick.AddListener(() =>
        {
            keyboard.gameObject.SetActive(!keyboard.gameObject.activeSelf);
        });
        handUI.onClick.AddListener(() =>
        {
            handTracker.enabled = !handTracker.enabled;
            if (handTracker.enabled)
            {
                testLog.text = "start hand";
                controllerTracker.enabled = false;
            }
            else
                testLog.text = "end hand";
        });
        controllerUI.onClick.AddListener(() =>
        {
            controllerTracker.enabled = !controllerTracker.enabled;
            if (controllerTracker.enabled)
            {
                testLog.text = "start controller";
                handTracker.enabled = false;
            }
            else
                testLog.text = "end controller";
        });
        frankaVR.onClick.AddListener(() =>
        {
            LoadRobot(frankaPrefab, true, true);
        });
        frankaReal.onClick.AddListener(() =>
        {
            LoadRobot(frankaPrefab, false, false);
        });
        flexivVR.onClick.AddListener(() =>
        {
            LoadRobot(flexivPrefab, true, true);
        });
        flexivReal.onClick.AddListener(() =>
        {
            LoadRobot(flexivPrefab, false, false);
        });
        ur10ShadowVR.onClick.AddListener(() =>
        {
            LoadRobot(ur10ShadowPrefab, true, true);
        });
        ur10ShadowReal.onClick.AddListener(() =>
        {
            LoadRobot(ur10ShadowPrefab, false, false);
        });
        shadowVR.onClick.AddListener(() =>
        {
            LoadRobot(shadowPrefab, false, true);
        });
        shadowReal.onClick.AddListener(() =>
        {
            LoadRobot(shadowPrefab, false, false);
        });
        bisionVR.onClick.AddListener(() =>
        {
            LoadBison(bisonPrefab, true, true);
        });
        bisionReal.onClick.AddListener(() =>
        {
            LoadBison(bisonPrefab, false, false);
        });
        shadowVRwithIK.onClick.AddListener(() =>
        {
            LoadRobot(shadowWithIKPrefab, false, true);
        });
        imageUI.onClick.AddListener(() =>
        {
            if (currentImage != null)
            {
                GameObject.DestroyImmediate(currentImage);
                currentImage = null;
            }
            else
            {
                currentImage = GameObject.Instantiate(imagePrefab);
                currentImage.transform.SetParent(WorldAlign.instance.transform);
            }
        });
    }

    void LoadRobot(ArticulationsController prefab, bool ik, bool sendOrReceive)
    {
        if (currentRobot != null)
        {
            GameObject.DestroyImmediate(currentRobot.gameObject);
            currentRobot = null;
        }
        else
        {
            currentRobot = GameObject.Instantiate(prefab);
            currentRobot.transform.SetParent(WorldAlign.instance.transform);
            if (ik)
                currentRobot.InitBioIK();
            currentRobot.sendOrReceive = sendOrReceive;
            currentRobot.SetPositionAndRotation(WorldAlign.instance.transform.position, WorldAlign.instance.transform.rotation);
        }
    }

    void LoadBison(ArticulationsController bison, bool ik, bool sendOrReceive)
    {
        if (currentRobot != null)
        {
            GameObject.DestroyImmediate(currentRobot.gameObject);
            currentRobot = null;
        }
        else
        {
            currentRobot = GameObject.Instantiate(bison);
            currentRobot.transform.SetParent(WorldAlign.instance.transform);
            foreach (var item in currentRobot.childs)
            {
                if(ik)
                {
                    item.InitBioIK();
                    item.sendOrReceive = sendOrReceive;
                }
            }
            currentRobot.SetPositionAndRotation(WorldAlign.instance.transform.position, WorldAlign.instance.transform.rotation);
        }
    }
}
