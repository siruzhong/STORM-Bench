using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using Newtonsoft.Json;

namespace StormDebug {
    public sealed class SafeFollowCamera : MonoBehaviour {
        public static UnityEngine.Object LoadSceneResource(string path) {
            var asset = Resources.Load(path);
            if (asset != null) return asset;
            var name = Path.GetFileNameWithoutExtension(path);
            foreach (var candidate in Resources.FindObjectsOfTypeAll<GameObject>()) {
                if (!candidate.scene.IsValid() || !candidate.activeInHierarchy) continue;
                if (String.Equals(candidate.name, name, StringComparison.OrdinalIgnoreCase)) {
                    Debug.Log("STORM scene prefab fallback: " + path + " -> " + candidate.name);
                    return candidate;
                }
            }
            return null;
        }
        private static Transform observer;
        private static SafeFollowCamera active;
        private static StreamWriter nativeClock;
        private static readonly Dictionary<int, int> nativeClockLast = new Dictionary<int, int>();
        private Vector3 worldPosition;
        private Quaternion worldRotation;
        private Vector3 velocity;
        private Vector3 springVelocity;
        private float yawVelocity;
        private float pitchVelocity;
        private float orbitSide;
        private Quaternion viewingHeading = Quaternion.identity;
        [Serializable] private sealed class ViewTarget { public float[] position; public float[] direction; }
        private readonly List<Bounds> keepouts = new List<Bounds>();
        [Serializable] private sealed class Keepout { public float[] center; public float[] size; }
        [Serializable] private sealed class KeepoutList { public Keepout[] obstacles; }
        private bool initialized;
        private bool captureStarted;
        private int sourceFrame = -1;
        [Serializable]
        private sealed class Rules {
            public string viewpoint = "third_person";
            public float max_speed_mps = 3.5f;
            public float max_turn_rate_dps = 120f;
            public float acceleration_mps2 = 10f;
            public float fov_degrees = 75f;
            public float focus_height_m = 1.5f;
            public float position_smooth_seconds = 0.30f;
            public float rotation_smooth_seconds = 0.22f;
            public float heading_smooth_seconds = 0.18f;
            public float goal_inertia_weight = 1.6f;
            public float[] position_min;
            public float[] position_max;
        }
        private Rules rules = new Rules();
        private static readonly List<Transform> characters = new List<Transform>();
        private Transform actor;
        private Vector3 desiredLocal;
        private Camera view;
        private StreamWriter trace;
        private string observerPosePath;
        private bool poseHeld;
        private Vector3 heldPosition;
        private Quaternion heldRotation;
        private const float Radius = 0.18f;
        private const float Clearance = 0.04f;

        public static Color InstanceColor(int instanceId) {
            // Preserve five bits per channel; the upstream encoder wraps at 512 IDs.
            uint value = (uint)instanceId & 32767u;
            return new Color(((value & 31u) * 8u + 4u) / 255f,
                             (((value >> 5) & 31u) * 8u + 4u) / 255f,
                             (((value >> 10) & 31u) * 8u + 4u) / 255f);
        }

        public static bool RecordCharacter(int characterIndex) {
            return characterIndex == 0 || Environment.GetEnvironmentVariable("STORM_OBSERVER_RENDER_ONLY") != "1";
        }

        public static void RecordNativeFrame(int frame, int fps, int characterIndex) {
            if (active == null) return;
            int previous;
            if (nativeClockLast.TryGetValue(characterIndex, out previous) && previous == frame) return;
            if (nativeClock == null) {
                string tracePath = Environment.GetEnvironmentVariable("STORM_CAMERA_TRACE");
                if (String.IsNullOrEmpty(tracePath)) return;
                nativeClock = new StreamWriter(Path.Combine(Path.GetDirectoryName(tracePath), "native_frame_clock.jsonl"));
                nativeClock.AutoFlush = true;
            }
            nativeClock.WriteLine(JsonConvert.SerializeObject(new {
                unity_frame = Time.frameCount, actor = characterIndex, source_frame = frame
            }));
            nativeClockLast[characterIndex] = frame;
        }

        public static int RecordingBudget(int requested) {
            // An idle character otherwise gets a shorter budget than the acting character.
            return Math.Max(requested, 1200);
        }

        public static void Attach(List<Camera> cameras, GameObject character) {
            characters.RemoveAll(item => item == null);
            if (!characters.Contains(character.transform)) characters.Add(character.transform);
            if (observer != null && observer != character.transform) return;
            observer = character.transform;
            if (nativeClock != null) { nativeClock.Dispose(); nativeClock = null; }
            nativeClockLast.Clear();
            NativeNavigation.Bind(observer);
            foreach (Camera camera in cameras) {
                if (!camera.name.Contains("REAR_OVERHEAD")) continue;
                var guard = camera.GetComponent<SafeFollowCamera>();
                if (guard != null) continue;
                guard = camera.gameObject.AddComponent<SafeFollowCamera>();
                guard.actor = character.transform;
                active = guard;
                string rulesPath = Environment.GetEnvironmentVariable("STORM_CAMERA_RULES");
                if (!String.IsNullOrEmpty(rulesPath)) JsonConvert.PopulateObject(File.ReadAllText(rulesPath), guard.rules);
                if (guard.rules.max_speed_mps <= 0f || guard.rules.max_turn_rate_dps <= 0f || guard.rules.acceleration_mps2 <= 0f)
                    throw new InvalidOperationException("Camera motion limits must be positive");
                if (guard.rules.position_smooth_seconds <= 0f || guard.rules.rotation_smooth_seconds <= 0f || guard.rules.heading_smooth_seconds <= 0f)
                    throw new InvalidOperationException("Camera smoothing times must be positive");
                if (guard.rules.viewpoint == "first_person")
                    foreach (Renderer renderer in character.GetComponentsInChildren<Renderer>()) renderer.enabled = false;
                guard.observerPosePath = Environment.GetEnvironmentVariable("STORM_OBSERVER_POSE");
                guard.view = camera;
                guard.desiredLocal = camera.transform.localPosition;
                string boxesPath = Environment.GetEnvironmentVariable("STORM_CAMERA_KEEPOUTS");
                if (!String.IsNullOrEmpty(boxesPath) && File.Exists(boxesPath)) {
                    var boxes = JsonConvert.DeserializeObject<KeepoutList>(File.ReadAllText(boxesPath));
                    foreach (var box in boxes.obstacles) {
                        Bounds bounds = new Bounds(new Vector3(box.center[0], box.center[1], box.center[2]),
                                                   new Vector3(box.size[0], box.size[1], box.size[2]));
                        bounds.Expand(2f * (Radius + Clearance));
                        guard.keepouts.Add(bounds);
                    }
                }
                camera.fieldOfView = guard.rules.fov_degrees;
                camera.nearClipPlane = 0.06f;
                string path = Environment.GetEnvironmentVariable("STORM_CAMERA_TRACE");
                if (!String.IsNullOrEmpty(path)) {
                    guard.trace = new StreamWriter(path, false);
                    guard.trace.AutoFlush = true;
                    guard.trace.WriteLine("frame,desired_distance,actual_distance,blocked,overlaps,collider,x,y,z");
                }
                Debug.Log("STORM_CAMERA_GUARD attached " + camera.name + " keepouts=" + guard.keepouts.Count + " speed=" + guard.rules.max_speed_mps + " turn=" + guard.rules.max_turn_rate_dps);
            }
        }

        private bool IsObstacle(Collider item, bool lineOfSight = false) {
            if (item == null || item.isTrigger) return false;
            if (rules.viewpoint == "first_person" && item.transform.IsChildOf(actor)) return false;
            // A passing person may occlude the view without forcing the camera into the observer.
            // Endpoint checks still keep the camera outside every other character's body.
            if (lineOfSight)
                foreach (Transform character in characters)
                    if (character != null && item.transform.IsChildOf(character)) return false;
            return true;
        }

        private bool Overlaps(Vector3 point) {
            if (rules.position_min != null && rules.position_max != null)
                for (int axis = 0; axis < 3; axis++)
                    if (point[axis] < rules.position_min[axis] || point[axis] > rules.position_max[axis]) return true;
            foreach (Bounds box in keepouts) if (box.Contains(point)) return true;
            foreach (Collider item in Physics.OverlapSphere(point, Radius, ~0, QueryTriggerInteraction.Ignore))
                if (IsObstacle(item)) return true;
            return false;
        }

        private Vector3 ClearPivot(Vector3 preferred) {
            if (!Overlaps(preferred)) return preferred;
            // Doors can swing into the usual anchor. Search near the head before casting.
            float[] heights = { 2.05f, 2.3f, 1.8f, 1.55f };
            float[] sides = { 0f, 0.25f, -0.25f, 0.5f, -0.5f };
            float[] depths = { 0f, -0.25f, -0.5f, 0.25f };
            foreach (float height in heights)
                foreach (float depth in depths)
                    foreach (float side in sides) {
                        Vector3 point = actor.TransformPoint(new Vector3(side, height, depth));
                        if (!Overlaps(point)) return point;
                    }
            return preferred;
        }

        private void HoldObserverPose() {
            if (String.IsNullOrEmpty(observerPosePath) || !File.Exists(observerPosePath)) {
                poseHeld = false;
                return;
            }
            if (!poseHeld) {
                heldPosition = actor.position;
                heldRotation = actor.rotation;
                poseHeld = true;
                Debug.Log("STORM_OBSERVER_HOLD " + actor.name + " " + heldPosition);
            }
            actor.SetPositionAndRotation(heldPosition, heldRotation);
        }

        private void LateUpdate() {
            NativeNavigation.Poll();
            // Other-character scripts can reset an idle character's transform.
            HoldObserverPose();
        }

        private float ClearDistance(Vector3 pivot, Vector3 offset, out string obstacle) {
            float length = offset.magnitude;
            Vector3 direction = offset.normalized;
            float safe = length;
            obstacle = "";
            foreach (RaycastHit hit in Physics.SphereCastAll(pivot, Radius, direction, length, ~0, QueryTriggerInteraction.Ignore)) {
                if (IsObstacle(hit.collider, true) && hit.distance - Clearance < safe) {
                    safe = Mathf.Max(0f, hit.distance - Clearance);
                    obstacle = hit.collider.name.Replace(',', '_');
                }
            }
            float boxDistance;
            Vector3 boxNormal;
            if (BoxHit(pivot, direction, safe, out boxDistance, out boxNormal)) {
                safe = Mathf.Max(0f, boxDistance - Clearance);
                obstacle = "camera_keepout";
            }
            while (safe > 0f && Overlaps(pivot + direction * safe)) safe = Mathf.Max(0f, safe - 0.025f);
            return safe;
        }

        private void StartCapture() {
            string marker = Environment.GetEnvironmentVariable("STORM_CAMERA_START");
            if (captureStarted || String.IsNullOrEmpty(marker) || !File.Exists(marker)) return;
            // Scene setup is outside the clip. Initialize once before its first saved frame.
            captureStarted = true;
            initialized = false;
            velocity = Vector3.zero;
            springVelocity = Vector3.zero;
            yawVelocity = pitchVelocity = 0f;
            orbitSide = 0f;
        }

        public static void BeginFrame(int frame, int fps, int characterIndex) {
            if (active == null || characterIndex != 0) return;
            active.StartCapture();
            if (active.sourceFrame != frame) {
                active.Advance(1f / Mathf.Max(1, fps));
                active.sourceFrame = frame;
            }
            active.ApplyPose();
        }

        public static void PrepareCamera(Camera camera, int characterIndex) {
            if (active == null || camera == null || characterIndex != 0) return;
            active.ApplyPose();
            camera.transform.SetPositionAndRotation(active.worldPosition, active.worldRotation);
            camera.fieldOfView = active.rules.fov_degrees;
        }

        public static void RecordPose(Camera camera, string filename, int characterIndex) {
            if (characterIndex != 0) return;
            if (filename.EndsWith("_normal.png"))
                File.WriteAllText(filename.Replace("_normal.png", "_camera.json"), PoseJson(camera));
            else if (filename.EndsWith("_seg_inst.png"))
                File.WriteAllText(filename.Replace("_seg_inst.png", "_seg_camera.json"), PoseJson(camera));
        }

        private static string PoseJson(Camera camera) {
            Vector3 p = camera.transform.position;
            Quaternion q = camera.transform.rotation;
            Vector3 a = observer.position;
            var actorPositions = new Dictionary<string, float[]>();
            for (int index = 0; index < characters.Count; index++) {
                if (characters[index] == null) continue;
                Vector3 position = characters[index].position;
                actorPositions[index.ToString(CultureInfo.InvariantCulture)] =
                    new float[] { position.x, position.y, position.z };
            }
            return JsonConvert.SerializeObject(new {
                position = new float[] { p.x, p.y, p.z },
                rotation = new float[] { q.x, q.y, q.z, q.w }, fov = camera.fieldOfView,
                observer_position = new float[] { a.x, a.y, a.z },
                actor_positions = actorPositions
            });
        }

        public static bool IsFirstPerson() {
            return active != null && active.rules.viewpoint == "first_person";
        }

        public static bool CanSeeObserver(Vector3 position) {
            if (active == null || !active.initialized) return false;
            Vector3 head = position + Vector3.up * 1.55f;
            Vector3 offset = active.worldPosition - head;
            string obstacle;
            return active.ClearDistance(head, offset, out obstacle) >= offset.magnitude - 0.02f;
        }

        public static bool CanTrack(Vector3 position, Vector3 forward) {
            if (active == null || forward.sqrMagnitude < 0.001f) return false;
            string obstacle;
            bool feasible;
            active.Goal(position + Vector3.up * 1.9f, position,
                Quaternion.LookRotation(forward, Vector3.up), out obstacle, out feasible);
            return feasible;
        }

        private Vector3 Goal(Vector3 pivot, Vector3 actorPosition, Quaternion heading,
                             out string obstacle, out bool feasible) {
            Vector3 nominal = actorPosition + heading * desiredLocal;
            if (rules.viewpoint == "first_person") {
                obstacle = "";
                feasible = !Overlaps(nominal);
                return nominal;
            }
            Vector3 best = nominal;
            float bestCost = float.PositiveInfinity;
            obstacle = "";
            // Score all clear shoulders against the previous pose; do not pick the first side.
            float[] angles = { 0f, 15f, -15f, 30f, -30f, 45f, -45f, 60f, -60f, 75f, -75f, 90f, -90f };
            foreach (float angle in angles) {
                Vector3 candidate = actorPosition + heading * Quaternion.Euler(0f, angle, 0f) * desiredLocal - pivot;
                string hit;
                float distance = ClearDistance(pivot, candidate, out hit);
                if (distance < 1.1f) continue;
                Vector3 point = pivot + candidate.normalized * distance;
                if (Overlaps(point)) continue;
                // A clear ray above the head can still leave the avatar behind furniture.
                Vector3 head = actorPosition + Vector3.up * 1.55f;
                string headObstacle;
                if (ClearDistance(head, point - head, out headObstacle) < (point - head).magnitude - 0.02f)
                    continue;
                float cost = (point - nominal).sqrMagnitude + (initialized ? rules.goal_inertia_weight * (point - worldPosition).sqrMagnitude : 0f);
                if (cost >= bestCost) continue;
                bestCost = cost;
                best = point;
                obstacle = hit;
            }
            feasible = !float.IsPositiveInfinity(bestCost);
            // An infeasible tracking goal must not cause a camera teleport.
            return float.IsPositiveInfinity(bestCost) && initialized ? worldPosition : best;
        }

        private bool BoxHit(Vector3 start, Vector3 direction, float length, out float distance, out Vector3 normal) {
            distance = length;
            normal = Vector3.zero;
            bool hit = false;
            foreach (Bounds box in keepouts) {
                float entry;
                if (!box.IntersectRay(new Ray(start, direction), out entry) || entry >= distance) continue;
                distance = Mathf.Max(0f, entry);
                Vector3 point = start + direction * distance;
                Vector3 offset = point - box.center;
                Vector3 edge = box.extents;
                float x = Mathf.Abs(Mathf.Abs(offset.x) - edge.x);
                float y = Mathf.Abs(Mathf.Abs(offset.y) - edge.y);
                float z = Mathf.Abs(Mathf.Abs(offset.z) - edge.z);
                normal = x <= y && x <= z ? new Vector3(Mathf.Sign(offset.x), 0f, 0f)
                       : y <= z ? new Vector3(0f, Mathf.Sign(offset.y), 0f)
                       : new Vector3(0f, 0f, Mathf.Sign(offset.z));
                hit = true;
            }
            return hit;
        }

        private Vector3 OrbitStep(Vector3 step, Vector3 goal, float dt) {
            Vector3 radial = worldPosition - actor.position;
            radial.y = 0f;
            Vector3 flatStep = new Vector3(step.x, 0f, step.z);
            float t = flatStep.sqrMagnitude > 0f ? Mathf.Clamp01(-Vector3.Dot(radial, flatStep) / flatStep.sqrMagnitude) : 0f;
            if ((radial + flatStep * t).magnitude >= 0.85f) {
                orbitSide = 0f;
                return step;
            }
            Vector3 target = goal - actor.position;
            target.y = 0f;
            if (orbitSide == 0f) orbitSide = Vector3.Cross(radial, target).y >= 0f ? 1f : -1f;
            float angle = orbitSide * Mathf.Min(80f * dt, step.magnitude / Mathf.Max(radial.magnitude, 0.85f) * Mathf.Rad2Deg);
            Vector3 around = Quaternion.Euler(0f, angle, 0f) * radial.normalized * Mathf.Max(0.9f, radial.magnitude);
            Vector3 result = around - radial;
            result.y = step.y;
            result = Vector3.ClampMagnitude(result, rules.max_speed_mps * dt);
            Vector3 travel = Sweep(worldPosition, result) - worldPosition;
            if (travel.magnitude < result.magnitude * 0.5f) {
                Vector3 opposite = Quaternion.Euler(0f, -angle, 0f) * radial.normalized * Mathf.Max(0.9f, radial.magnitude) - radial;
                opposite.y = step.y;
                opposite = Vector3.ClampMagnitude(opposite, rules.max_speed_mps * dt);
                if ((Sweep(worldPosition, opposite) - worldPosition).magnitude > travel.magnitude + 0.01f) {
                    orbitSide *= -1f;
                    result = opposite;
                }
            }
            return result;
        }

        private Vector3 Sweep(Vector3 start, Vector3 step) {
            if (step.sqrMagnitude < 0.00000001f) return start;
            Vector3 result = start;
            Vector3 remaining = step;
            for (int pass = 0; pass < 3 && remaining.sqrMagnitude > 0.00000001f; pass++) {
                float length = remaining.magnitude;
                Vector3 contactNormal = Vector3.zero;
                bool blocked = false;
                float distance = length;
                foreach (RaycastHit hit in Physics.SphereCastAll(result, Radius, remaining.normalized, length + Clearance, ~0, QueryTriggerInteraction.Ignore)) {
                    if (!IsObstacle(hit.collider) || hit.distance >= distance) continue;
                    distance = Mathf.Max(0f, hit.distance - Clearance);
                    contactNormal = hit.normal;
                    blocked = true;
                }
                float boxDistance;
                Vector3 boxNormal;
                if (BoxHit(result, remaining.normalized, distance, out boxDistance, out boxNormal)) {
                    distance = Mathf.Max(0f, boxDistance - Clearance);
                    contactNormal = boxNormal;
                    blocked = true;
                }
                Vector3 travel = remaining.normalized * distance;
                result += travel;
                if (!blocked) break;
                remaining = Vector3.ProjectOnPlane(remaining - travel, contactNormal);
            }
            if (Overlaps(result)) return start;
            // A slide cannot exceed the original per-frame travel budget.
            Vector3 direct = Vector3.ClampMagnitude(result - start, step.magnitude);
            float safe = direct.magnitude;
            if (safe < 0.00001f) return start;
            foreach (RaycastHit hit in Physics.SphereCastAll(start, Radius, direct.normalized, safe + Clearance, ~0, QueryTriggerInteraction.Ignore))
                if (IsObstacle(hit.collider)) safe = Mathf.Min(safe, Mathf.Max(0f, hit.distance - Clearance));
            float boxEntry;
            Vector3 boxFace;
            if (BoxHit(start, direct.normalized, safe, out boxEntry, out boxFace)) safe = Mathf.Max(0f, boxEntry - Clearance);
            return start + direct.normalized * safe;
        }

        private void Advance(float dt) {
            HoldObserverPose();
            Physics.SyncTransforms();
            bool firstPerson = rules.viewpoint == "first_person";
            Vector3 pivot = firstPerson ? actor.position + Vector3.up * desiredLocal.y
                : ClearPivot(actor.TransformPoint(new Vector3(0f, 1.9f, 0f)));
            float viewPitch = 0f;
            Quaternion targetHeading = actor.rotation;
            string viewPath = Environment.GetEnvironmentVariable("STORM_CAMERA_VIEW_TARGET");
            if (!String.IsNullOrEmpty(viewPath) && File.Exists(viewPath)) {
                string text = File.ReadAllText(viewPath);
                var target = JsonConvert.DeserializeObject<ViewTarget>(text);
                Vector3 flat = target.direction != null
                    ? new Vector3(target.direction[0], 0f, target.direction[2])
                    : new Vector3(target.position[0], actor.position.y, target.position[2]) - actor.position;
                if (flat.sqrMagnitude > 0.04f) targetHeading = Quaternion.LookRotation(flat, Vector3.up);
                if (firstPerson && target.position != null)
                    viewPitch = Mathf.Clamp(Mathf.Atan2(target.position[1] - pivot.y,
                        Mathf.Max(flat.magnitude, 0.2f)) * Mathf.Rad2Deg, -50f, 20f);
            }
            viewingHeading = initialized
                ? Quaternion.Slerp(viewingHeading, targetHeading, 1f - Mathf.Exp(-dt / rules.heading_smooth_seconds))
                : targetHeading;
            string obstacle;
            bool feasibleGoal;
            Vector3 goal = Goal(pivot, actor.position, viewingHeading, out obstacle, out feasibleGoal);
            Vector3 focus = actor.position + new Vector3(0f, rules.focus_height_m, 0f) + viewingHeading * Vector3.forward * 0.45f;
            if (firstPerson)
                focus = pivot + viewingHeading * Vector3.forward * 3f
                    + Vector3.up * (Mathf.Tan(viewPitch * Mathf.Deg2Rad) * 3f);
            if (!initialized) {
                worldPosition = goal;
                worldRotation = Quaternion.LookRotation(focus - goal, Vector3.up);
                initialized = true;
            } else {
                // Damping runs on the recording clock and keeps its state across actions.
                Vector3 requested = Vector3.SmoothDamp(worldPosition, goal, ref springVelocity,
                    rules.position_smooth_seconds, rules.max_speed_mps, dt);
                Vector3 wanted = Vector3.ClampMagnitude((requested - worldPosition) / dt, rules.max_speed_mps);
                Vector3 previous = worldPosition;
                Vector3 step = firstPerson ? wanted * dt : OrbitStep(wanted * dt, goal, dt);
                Vector3 candidate = previous + step;
                float previousError = Quaternion.Angle(worldRotation, Quaternion.LookRotation(focus - previous, Vector3.up));
                float candidateError = Quaternion.Angle(worldRotation, Quaternion.LookRotation(focus - candidate, Vector3.up));
                // Use a continuous framing correction instead of repeated half-step braking.
                // Movement that brings the observer back toward the image center remains allowed.
                if (!firstPerson && candidateError > 12f && candidateError > previousError + 0.001f) {
                    float fraction = Mathf.Clamp01((12f - previousError) / (candidateError - previousError));
                    step *= fraction;
                }
                // Apply acceleration limits after orbit and framing corrections.
                // Collision sweeps remain the final constraint on physical movement.
                wanted = Vector3.ClampMagnitude(step / dt, rules.max_speed_mps);
                velocity = Vector3.MoveTowards(velocity, wanted, rules.acceleration_mps2 * dt);
                worldPosition = Sweep(previous, velocity * dt);
                velocity = (worldPosition - previous) / dt;
                // Collision correction takes precedence over the spring's requested position.
                if ((worldPosition - requested).sqrMagnitude > 0.000001f) springVelocity = velocity;
                Quaternion look = Quaternion.LookRotation(focus - worldPosition, Vector3.up);
                Vector3 currentAngles = worldRotation.eulerAngles;
                Vector3 targetAngles = look.eulerAngles;
                float yaw = Mathf.SmoothDampAngle(currentAngles.y, targetAngles.y, ref yawVelocity,
                    rules.rotation_smooth_seconds, rules.max_turn_rate_dps, dt);
                float pitch = Mathf.SmoothDampAngle(currentAngles.x, targetAngles.x, ref pitchVelocity,
                    rules.rotation_smooth_seconds, rules.max_turn_rate_dps, dt);
                Quaternion eased = Quaternion.Euler(pitch, yaw, 0f);
                worldRotation = Quaternion.RotateTowards(worldRotation, eased, rules.max_turn_rate_dps * dt);
            }
            ApplyPose();
            if (trace != null) {
                float desired = (actor.position + viewingHeading * desiredLocal - pivot).magnitude;
                float actual = (worldPosition - pivot).magnitude;
                trace.WriteLine(String.Format(CultureInfo.InvariantCulture, "{0},{1:F5},{2:F5},{3},{4},{5},{6:F5},{7:F5},{8:F5}",
                    Time.frameCount, desired, actual, actual < desired - 0.001f ? 1 : 0,
                    Overlaps(worldPosition) ? 1 : 0, obstacle, worldPosition.x, worldPosition.y, worldPosition.z));
            }
        }

        private void ApplyPose() {
            if (initialized) {
                transform.SetPositionAndRotation(worldPosition, worldRotation);
                view.fieldOfView = rules.fov_degrees;
            }
        }

        private void OnPreCull() {
            StartCapture();
            // HTTP snapshots and segmentation renders must not advance the camera clock.
            if (!initialized) Advance(1f / 20f);
            ApplyPose();
            string path = Environment.GetEnvironmentVariable("STORM_CAMERA_TRACE");
            if (!String.IsNullOrEmpty(path)) File.WriteAllText(path + ".pose.json", PoseJson(view));
        }

        private void OnDestroy() {
            if (trace != null) { trace.Dispose(); trace = null; }
        }
    }
}
