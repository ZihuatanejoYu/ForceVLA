using System.Collections.Generic;
using System.Linq;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
using Unity.Robotics.UrdfImporter;
using Unity.Robotics.UrdfImporter.Control;
#endif

public static class GlobalUtility
{
#if UNITY_EDITOR
    [MenuItem("Tools/NormalizeArticulations")]
    public static void NormalizeArticulations()
    {
        NormalizeArticulations(Selection.activeGameObject);
    }

    public static ArticulationsController NormalizeArticulations(GameObject root)
    {
        ArticulationsController sync = root.AddComponent<ArticulationsController>();

        // Remove URDFImporter Scripts
        UrdfPlugins urdfPlugins = root.GetComponentInChildren<UrdfPlugins>();
        if (urdfPlugins != null)
            GameObject.DestroyImmediate(urdfPlugins.gameObject);
        Controller controller = root.GetComponentInChildren<Controller>();
        if (controller != null)
            GameObject.DestroyImmediate(controller);
        UrdfRobot urdfRobot = root.GetComponentInChildren<UrdfRobot>();
        if (urdfRobot != null)
            GameObject.DestroyImmediate(urdfRobot);
        UrdfLink[] urdfLinks = root.GetComponentsInChildren<UrdfLink>();
        foreach (var urdfLink in urdfLinks)
        {
            GameObject.DestroyImmediate(urdfLink);
        }
        UrdfInertial[] urdfInertial = root.GetComponentsInChildren<UrdfInertial>();
        foreach (var item in urdfInertial)
        {
            GameObject.DestroyImmediate(item);
        }

        UrdfVisual[] urdfVisual = root.GetComponentsInChildren<UrdfVisual>();
        foreach (var item in urdfVisual)
        {
            GameObject.DestroyImmediate(item);
        }

        UrdfCollision[] urdfCollision = root.GetComponentsInChildren<UrdfCollision>();
        foreach (var item in urdfCollision)
        {
            GameObject.DestroyImmediate(item);
        }

        // Add basic script for root node
        IgnoreSelfCollision ign = root.GetComponent<IgnoreSelfCollision>() ?? root.AddComponent<IgnoreSelfCollision>();
        if (root.transform.GetChild(0).GetComponent<ArticulationBody>() == null)
            root.transform.GetChild(0).gameObject.AddComponent<ArticulationBody>();

        // Add RFUniverse scripts
        ArticulationBody[] articulationBodies = root.GetComponentsInChildren<ArticulationBody>();
        foreach (var body in articulationBodies)
        {
            ArticulationUnit unit = body.GetUnit();
            UrdfJoint joint = body.GetComponent<UrdfJoint>();
            unit.jointName = body.name;
            if (joint)
            {
                unit.jointName = joint.jointName;
                if (!string.IsNullOrEmpty(joint.minicJointParentName))
                {
                    ArticulationBody mimicParent = articulationBodies.FirstOrDefault(s => s.GetComponent<UrdfJoint>()?.jointName == joint.minicJointParentName);
                    unit.mimicParent = mimicParent;
                    unit.mimicMultiplier = joint.multiplier;
                    unit.mimicOffset = joint.offset;
                }
            }
            if (body.isRoot)
                body.immovable = true;
            body.useGravity = false;

            body.linearDamping = 0.05f;
            body.angularDamping = 0.05f;
            body.jointFriction = 0.05f;

            var xDrive = body.xDrive;
            xDrive.stiffness = 100000;
            xDrive.damping = 9000;
            xDrive.forceLimit = float.MaxValue;
            body.xDrive = xDrive;

            var yDrive = body.yDrive;
            yDrive.stiffness = 100000;
            yDrive.damping = 9000;
            yDrive.forceLimit = float.MaxValue;
            body.yDrive = yDrive;

            var zDrive = body.zDrive;
            zDrive.stiffness = 100000;
            zDrive.damping = 9000;
            zDrive.forceLimit = float.MaxValue;
            body.zDrive = zDrive;

            List<Transform> renders = GetChildComponentFilter<ArticulationBody, Renderer>(body).Select((s) => s.transform).ToList();
            if (renders.Count == 0)
                renders.Add(new GameObject("None").transform);
            foreach (var item in renders)
            {
                item.SetParent(body.transform);
                item.SetSiblingIndex(0);
            }
            List<Collider> colliders = GetChildComponentFilter<ArticulationBody, Collider>(body);
            for (int i = 0; i < colliders.Count; i++)
            {
                Collider collider = colliders[i];
                collider.name = "Collider";
                int index = Mathf.Min(renders.Count - 1, i);
                if (index >= 0 && index < renders.Count)
                    collider.transform.parent = renders[index];
                else
                    collider.transform.parent = body.transform;
            }
        }

        UrdfVisuals[] urdfVisuals = root.GetComponentsInChildren<UrdfVisuals>();
        foreach (var item in urdfVisuals)
        {
            GameObject.DestroyImmediate(item.gameObject);
        }
        UrdfCollisions[] urdfCollisions = root.GetComponentsInChildren<UrdfCollisions>();
        foreach (var item in urdfCollisions)
        {
            GameObject.DestroyImmediate(item.gameObject);
        }
        UrdfJoint[] urdfJoints = root.GetComponentsInChildren<UrdfJoint>();
        foreach (var item in urdfJoints)
        {
            GameObject.DestroyImmediate(item);
        }

        sync.InitJoints();
        return sync;
    }
#endif
    public static ArticulationUnit GetUnit(this ArticulationBody body)
    {
        if (body.TryGetComponent(out ArticulationUnit unit))
            return unit;
        else
            return body.gameObject.AddComponent<ArticulationUnit>();
    }

    public static List<TResult> GetChildComponentFilter<TParent, TResult>(TParent parent) where TParent : Component where TResult : Component
    {
        List<TResult> components = new List<TResult>();
        foreach (var item in parent.GetComponentsInChildren<TResult>())
        {
            if (item.GetComponentInParent<TParent>() == parent)
                components.Add(item);
        }
        return components;
    }
}
