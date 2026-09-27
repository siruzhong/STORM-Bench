using System;
using System.Linq;
using Mono.Cecil;
using Mono.Cecil.Cil;

class PatchCamera {
    static void Main(string[] args) {
        var resolver = new DefaultAssemblyResolver();
        resolver.AddSearchDirectory(System.IO.Path.GetDirectoryName(args[0]));
        var parameters = new ReaderParameters { AssemblyResolver = resolver };
        using (var source = AssemblyDefinition.ReadAssembly(args[0], parameters))
        using (var addon = AssemblyDefinition.ReadAssembly(args[1], parameters)) {
            if (source.MainModule.AssemblyReferences.Any(a => a.Name == "StormCameraGuard"))
                throw new InvalidOperationException("Assembly is already patched");
            var type = source.MainModule.Types.Single(t => t.FullName == "StoryGenerator.Utilities.CameraExpander");
            var method = type.Methods.Single(m => m.Name == "AddCharacterCameras");
            var hook = addon.MainModule.Types.Single(t => t.FullName == "StormDebug.SafeFollowCamera")
                            .Methods.Single(m => m.Name == "Attach");
            var imported = source.MainModule.ImportReference(hook);
            var il = method.Body.GetILProcessor();
            foreach (var end in method.Body.Instructions.Where(i => i.OpCode == OpCodes.Ret).ToArray()) {
                il.InsertBefore(end, il.Create(OpCodes.Dup));
                il.InsertBefore(end, il.Create(OpCodes.Ldarg_0));
                il.InsertBefore(end, il.Create(OpCodes.Call, imported));
            }
            var recorder = source.MainModule.Types.Single(t => t.FullName == "StoryGenerator.Recording.Recorder");
            var guardType = addon.MainModule.Types.Single(t => t.FullName == "StormDebug.SafeFollowCamera");
            var navigation = addon.MainModule.Types.Single(t => t.FullName == "StormDebug.NativeNavigation");
            var characterControl = source.MainModule.Types.Single(t => t.FullName == "StoryGenerator.CharacterControl");
            if (args.Length > 3 && args[3] == "--observer-walk-pacing") {
                var walk = characterControl.Methods.Single(m => m.Name == "walkOrRunTo"
                    && m.Parameters.Count == 4 && m.Parameters[1].ParameterType.FullName == "UnityEngine.Vector3");
                var pace = source.MainModule.ImportReference(navigation.Methods.Single(m => m.Name == "PaceWalk"));
                var walkIl = walk.Body.GetILProcessor();
                foreach (var end in walk.Body.Instructions.Where(i => i.OpCode == OpCodes.Ret).ToArray()) {
                    walkIl.InsertBefore(end, walkIl.Create(OpCodes.Ldarg_0));
                    walkIl.InsertBefore(end, walkIl.Create(OpCodes.Call, pace));
                }
            }
            var selectPosition = characterControl.Methods.Single(m => m.Name == "SelectPosAndLookAt");
            var navigationIl = selectPosition.Body.GetILProcessor();
            var navigationEntry = selectPosition.Body.Instructions[0];
            navigationIl.InsertBefore(navigationEntry, navigationIl.Create(OpCodes.Ldarg_0));
            navigationIl.InsertBefore(navigationEntry, navigationIl.Create(OpCodes.Call,
                source.MainModule.ImportReference(navigation.Methods.Single(m => m.Name == "HasTarget"))));
            navigationIl.InsertBefore(navigationEntry, navigationIl.Create(OpCodes.Brfalse, navigationEntry));
            navigationIl.InsertBefore(navigationEntry, navigationIl.Create(OpCodes.Ldarg_0));
            navigationIl.InsertBefore(navigationEntry, navigationIl.Create(OpCodes.Ldarg, selectPosition.Parameters[2]));
            navigationIl.InsertBefore(navigationEntry, navigationIl.Create(OpCodes.Ldarg, selectPosition.Parameters[3]));
            navigationIl.InsertBefore(navigationEntry, navigationIl.Create(OpCodes.Ldarg, selectPosition.Parameters[4]));
            navigationIl.InsertBefore(navigationEntry, navigationIl.Create(OpCodes.Call,
                source.MainModule.ImportReference(navigation.Methods.Single(m => m.Name == "SelectTarget"))));
            navigationIl.InsertBefore(navigationEntry, navigationIl.Create(OpCodes.Ret));
            var save = recorder.Methods.Single(m => m.Name == "SaveRenderedFromCam");
            var saveIl = save.Body.GetILProcessor();
            var saveFirst = save.Body.Instructions[0];
            if (save.ReturnType.FullName != "System.Void")
                throw new InvalidOperationException("Unexpected recorder signature");
            var nativeClockHook = source.MainModule.ImportReference(guardType.Methods.Single(m => m.Name == "RecordNativeFrame"));
            saveIl.InsertBefore(saveFirst, saveIl.Create(OpCodes.Ldarg_0));
            saveIl.InsertBefore(saveFirst, saveIl.Create(OpCodes.Ldfld, recorder.Fields.Single(f => f.Name == "frameNum")));
            saveIl.InsertBefore(saveFirst, saveIl.Create(OpCodes.Ldarg_0));
            saveIl.InsertBefore(saveFirst, saveIl.Create(OpCodes.Call, recorder.Methods.Single(m => m.Name == "get_FrameRate")));
            saveIl.InsertBefore(saveFirst, saveIl.Create(OpCodes.Ldarg_0));
            saveIl.InsertBefore(saveFirst, saveIl.Create(OpCodes.Call, recorder.Methods.Single(m => m.Name == "get_charIdx")));
            saveIl.InsertBefore(saveFirst, saveIl.Create(OpCodes.Call, nativeClockHook));
            var selected = source.MainModule.ImportReference(guardType.Methods.Single(m => m.Name == "RecordCharacter"));
            var renderEntry = saveIl.Create(OpCodes.Nop);
            saveIl.InsertBefore(saveFirst, saveIl.Create(OpCodes.Ldarg_0));
            saveIl.InsertBefore(saveFirst, saveIl.Create(OpCodes.Call, recorder.Methods.Single(m => m.Name == "get_charIdx")));
            saveIl.InsertBefore(saveFirst, saveIl.Create(OpCodes.Call, selected));
            saveIl.InsertBefore(saveFirst, saveIl.Create(OpCodes.Brtrue, renderEntry));
            saveIl.InsertBefore(saveFirst, saveIl.Create(OpCodes.Ret));
            saveIl.InsertBefore(saveFirst, renderEntry);
            var begin = source.MainModule.ImportReference(guardType.Methods.Single(m => m.Name == "BeginFrame"));
            saveIl.InsertBefore(saveFirst, saveIl.Create(OpCodes.Ldarg_0));
            saveIl.InsertBefore(saveFirst, saveIl.Create(OpCodes.Ldfld, recorder.Fields.Single(f => f.Name == "frameNum")));
            saveIl.InsertBefore(saveFirst, saveIl.Create(OpCodes.Ldarg_0));
            saveIl.InsertBefore(saveFirst, saveIl.Create(OpCodes.Call, recorder.Methods.Single(m => m.Name == "get_FrameRate")));
            saveIl.InsertBefore(saveFirst, saveIl.Create(OpCodes.Ldarg_0));
            saveIl.InsertBefore(saveFirst, saveIl.Create(OpCodes.Call, recorder.Methods.Single(m => m.Name == "get_charIdx")));
            saveIl.InsertBefore(saveFirst, saveIl.Create(OpCodes.Call, begin));
            var renderCall = save.Body.Instructions.Single(i => i.Operand is MethodReference && ((MethodReference)i.Operand).FullName == "System.Void UnityEngine.Camera::Render()");
            var prepare = source.MainModule.ImportReference(guardType.Methods.Single(m => m.Name == "PrepareCamera"));
            saveIl.InsertBefore(renderCall, saveIl.Create(OpCodes.Dup));
            saveIl.InsertBefore(renderCall, saveIl.Create(OpCodes.Ldarg_0));
            saveIl.InsertBefore(renderCall, saveIl.Create(OpCodes.Call, recorder.Methods.Single(m => m.Name == "get_charIdx")));
            saveIl.InsertBefore(renderCall, saveIl.Create(OpCodes.Call, prepare));
            var write = save.Body.Instructions.Single(i => i.Operand is MethodReference && ((MethodReference)i.Operand).FullName.Contains("System.IO.File::WriteAllBytes"));
            var next = write.Next;
            var poseHook = source.MainModule.ImportReference(guardType.Methods.Single(m => m.Name == "RecordPose"));
            saveIl.InsertBefore(next, saveIl.Create(OpCodes.Ldloc_0));
            saveIl.InsertBefore(next, saveIl.Create(OpCodes.Ldloc, save.Body.Variables[4]));
            saveIl.InsertBefore(next, saveIl.Create(OpCodes.Ldarg_0));
            saveIl.InsertBefore(next, saveIl.Create(OpCodes.Call, recorder.Methods.Single(m => m.Name == "get_charIdx")));
            saveIl.InsertBefore(next, saveIl.Create(OpCodes.Call, poseHook));
            var setter = recorder.Methods.Single(m => m.Name == "set_MaxFrameNumber");
            var budget = addon.MainModule.Types.Single(t => t.FullName == "StormDebug.SafeFollowCamera")
                              .Methods.Single(m => m.Name == "RecordingBudget");
            var setterIl = setter.Body.GetILProcessor();
            var first = setter.Body.Instructions[0];
            setterIl.InsertBefore(first, setterIl.Create(OpCodes.Ldarg_1));
            setterIl.InsertBefore(first, setterIl.Create(OpCodes.Call, source.MainModule.ImportReference(budget)));
            setterIl.InsertBefore(first, setterIl.Create(OpCodes.Starg_S, setter.Parameters[0]));
            var encoder = source.MainModule.Types.Single(t => t.FullName == "StoryGenerator.Recording.ColorEncoding")
                                .Methods.Single(m => m.Name == "EncodeIDAsColor");
            var colorHook = addon.MainModule.Types.Single(t => t.FullName == "StormDebug.SafeFollowCamera")
                                 .Methods.Single(m => m.Name == "InstanceColor");
            encoder.Body.Instructions.Clear();
            encoder.Body.Variables.Clear();
            encoder.Body.ExceptionHandlers.Clear();
            encoder.Body.InitLocals = false;
            var colorIl = encoder.Body.GetILProcessor();
            colorIl.Append(colorIl.Create(OpCodes.Ldarg_0));
            colorIl.Append(colorIl.Create(OpCodes.Call, source.MainModule.ImportReference(colorHook)));
            colorIl.Append(colorIl.Create(OpCodes.Ret));
            var expander = source.MainModule.Types.Single(t => t.FullName == "StoryGenerator.Utilities.SceneExpander");
            var resourceHook = source.MainModule.ImportReference(guardType.Methods.Single(m => m.Name == "LoadSceneResource"));
            int replaced = 0;
            foreach (var expansionMethod in expander.Methods.Where(m => m.HasBody)) {
                foreach (var instruction in expansionMethod.Body.Instructions) {
                    var called = instruction.Operand as MethodReference;
                    if (called != null && called.FullName == "UnityEngine.Object UnityEngine.Resources::Load(System.String)") {
                        instruction.Operand = resourceHook;
                        replaced++;
                    }
                }
            }
            if (replaced == 0) throw new InvalidOperationException("No scene resource loaders found");
            Console.WriteLine("Patched scene resource loaders: " + replaced);
            source.Write(args[2]);
            Console.WriteLine("Camera guard patched into " + args[2]);
        }
    }
}
