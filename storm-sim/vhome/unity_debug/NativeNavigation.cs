using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.AI;
using Newtonsoft.Json;

namespace StormDebug {
    public sealed class NativeNavigationPump : MonoBehaviour {
        private void Update() { NativeNavigation.Poll(); }
    }
    public static class NativeNavigation {
        private static Transform observer;
        private static string directory;
        private static string handledRequest;
        private static string handledCameraRequest;
        [Serializable] private sealed class Request {
            public string id;
            public float[][] points;
            public int[][] edges;
            public float sample_radius = 0.35f;
            public SightTarget[] sight_targets;
            public float camera_arm = 1.7f;
            public float camera_height = 2.2f;
            public float camera_side = 0.35f;
        }
        [Serializable] private sealed class SightTarget {
            public int id;
            public float[] center;
            public float[] size;
        }
        [Serializable] private sealed class CameraRouteRequest {
            public string id;
            public float[][][] routes;
        }
        private static void PollCameraRoutes() {
            string input = Path.Combine(directory, "camera_routes_request.json");
            if (!File.Exists(input)) return;
            var request = JsonConvert.DeserializeObject<CameraRouteRequest>(File.ReadAllText(input));
            if (request.id == handledCameraRequest) return;
            var failures = new List<int>();
            foreach (var route in request.routes) {
                int blocked = 0;
                for (int index = 1; index < route.Length; index++) {
                    Vector3 start = Vector(route[index - 1]), end = Vector(route[index]);
                    int steps = Math.Max(1, Mathf.CeilToInt(Vector3.Distance(start, end) / 0.3f));
                    for (int step = 1; step <= steps; step++)
                        if (!SafeFollowCamera.CanSeeObserver(Vector3.Lerp(start, end, (float)step / steps))) blocked++;
                }
                failures.Add(blocked);
            }
            string output = Path.Combine(directory, "camera_routes_response.json");
            File.WriteAllText(output + ".tmp", JsonConvert.SerializeObject(new { id = request.id, blocked = failures }));
            if (File.Exists(output)) File.Delete(output);
            File.Move(output + ".tmp", output);
            handledCameraRequest = request.id;
        }
        [Serializable] private sealed class Target {
            public float[] position;
            public float[] look_at;
            public float animation_speed_scale = 1.0f;
        }
        public static IEnumerator PaceWalk(IEnumerator movement, Component character) {
            Animator animator = null;
            float originalSpeed = 1.0f;
            if (HasTarget(character)) {
                var target = JsonConvert.DeserializeObject<Target>(File.ReadAllText(
                    Path.Combine(directory, "observer_destination.json")));
                animator = character.GetComponent<Animator>();
                if (animator != null) {
                    originalSpeed = animator.speed;
                    animator.speed = originalSpeed * Mathf.Clamp(target.animation_speed_scale, 0.45f, 1.0f);
                }
            }
            try {
                while (movement.MoveNext()) yield return movement.Current;
            } finally {
                if (animator != null) animator.speed = originalSpeed;
                var disposable = movement as IDisposable;
                if (disposable != null) disposable.Dispose();
            }
        }
        private static Vector3 Vector(float[] value) {
            return new Vector3(value[0], value[1], value[2]);
        }
        private static float[] Array(Vector3 value) {
            return new float[] { value.x, value.y, value.z };
        }
        public static void Bind(Transform character) {
            observer = character;
            handledRequest = null;
            handledCameraRequest = null;
            var trace = Environment.GetEnvironmentVariable("STORM_CAMERA_TRACE");
            directory = String.IsNullOrEmpty(trace) ? null : Path.GetDirectoryName(trace);
            if (character.GetComponent<NativeNavigationPump>() == null)
                character.gameObject.AddComponent<NativeNavigationPump>();
        }
        public static void Poll() {
            if (directory == null || observer == null) return;
            try { PollCameraRoutes(); }
            catch (Exception error) { Debug.LogError("STORM camera route probe: " + error.Message); }
            string input = Path.Combine(directory, "navigation_request.json");
            if (!File.Exists(input)) return;
            try {
                var request = JsonConvert.DeserializeObject<Request>(File.ReadAllText(input));
                if (request.id == handledRequest) return;
                var positions = new Vector3[request.points.Length];
                var valid = new bool[positions.Length];
                var points = new List<object>();
                for (int index = 0; index < positions.Length; index++) {
                    NavMeshHit hit;
                    valid[index] = NavMesh.SamplePosition(Vector(request.points[index]), out hit,
                                                          request.sample_radius, NavMesh.AllAreas);
                    positions[index] = hit.position;
                    points.Add(new { index = index, valid = valid[index], position = Array(hit.position) });
                }
                var paths = new List<object>();
                foreach (var edge in request.edges) {
                    int from = edge[0], to = edge[1];
                    var path = new NavMeshPath();
                    bool complete = valid[from] && valid[to] && NavMesh.CalculatePath(
                        positions[from], positions[to], NavMesh.AllAreas, path)
                        && path.status == NavMeshPathStatus.PathComplete;
                    float length = 0f;
                    var corners = new List<float[]>();
                    if (complete) {
                        for (int index = 0; index < path.corners.Length; index++) {
                            corners.Add(Array(path.corners[index]));
                            if (index > 0) length += Vector3.Distance(path.corners[index - 1], path.corners[index]);
                        }
                    }
                    paths.Add(new { from = from, to = to, complete = complete,
                                    length = length, corners = corners });
                }
                var sight = new List<object>();
                if (request.sight_targets != null) {
                    foreach (var target in request.sight_targets) {
                        Vector3 center = Vector(target.center);
                        Bounds bounds = new Bounds(center, Vector(target.size));
                        bounds.Expand(0.05f);
                        for (int index = 0; index < positions.Length; index++) {
                            if (!valid[index]) continue;
                            Vector3 forward = center - positions[index];
                            forward.y = 0f;
                            forward.Normalize();
                            Vector3 right = Vector3.Cross(Vector3.up, forward);
                            Vector3 camera = positions[index] - forward * request.camera_arm
                                + right * request.camera_side + Vector3.up * request.camera_height;
                            bool clearCamera = true;
                            foreach (var collider in Physics.OverlapSphere(camera, 0.18f, ~0, QueryTriggerInteraction.Ignore))
                                if (!collider.transform.IsChildOf(observer)) clearCamera = false;
                            int visible = 0;
                            foreach (float vertical in new float[] { -0.3f, 0f, 0.3f }) {
                                Vector3 sample = center + Vector3.up * bounds.size.y * vertical;
                                Vector3 offset = sample - camera;
                                bool clear = clearCamera;
                                foreach (var hit in Physics.RaycastAll(camera, offset.normalized, offset.magnitude, ~0, QueryTriggerInteraction.Ignore)) {
                                    if (hit.collider.transform.IsChildOf(observer)) continue;
                                    if (!bounds.Contains(hit.point)) { clear = false; break; }
                                }
                                if (clear) visible++;
                            }
                            sight.Add(new { point = index, target_id = target.id, clear_rays = visible,
                                observer_camera_clear = SafeFollowCamera.CanTrack(positions[index], forward),
                                hidden_camera_clear = SafeFollowCamera.CanTrack(positions[index], -forward) });
                        }
                    }
                }
                string output = Path.Combine(directory, "navigation_response.json");
                File.WriteAllText(output + ".tmp", JsonConvert.SerializeObject(
                    new { id = request.id, points = points, paths = paths, sight = sight, camera_route_probe = !SafeFollowCamera.IsFirstPerson() }));
                if (File.Exists(output)) File.Delete(output);
                File.Move(output + ".tmp", output);
                handledRequest = request.id;
            } catch (Exception error) {
                Debug.LogError("STORM navigation probe: " + error.Message);
            }
        }
        public static bool HasTarget(Component character) {
            return observer != null && character.transform == observer && directory != null
                && File.Exists(Path.Combine(directory, "observer_destination.json"));
        }
        public static bool SelectTarget(Component character, out Vector3 position,
                                        out Vector3? lookAt, out NavMeshPath path) {
            position = character.transform.position;
            lookAt = null;
            path = new NavMeshPath();
            var target = JsonConvert.DeserializeObject<Target>(File.ReadAllText(
                Path.Combine(directory, "observer_destination.json")));
            NavMeshHit hit;
            if (!NavMesh.SamplePosition(Vector(target.position), out hit, 0.2f, NavMesh.AllAreas)) return false;
            position = hit.position;
            if (target.look_at != null) lookAt = Vector(target.look_at);
            return NavMesh.CalculatePath(character.transform.position, position, NavMesh.AllAreas, path)
                && path.status == NavMeshPathStatus.PathComplete;
        }
    }
}
