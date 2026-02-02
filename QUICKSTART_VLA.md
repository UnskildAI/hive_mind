# VLA Quick Start Guide

## 🚀 Get Started in 3 Steps

### 1. Test GPU Setup

```bash
cd /home/mecha/hive_mind
python scripts/test_vla_inference.py --gpu-info
```

### 2. Test VLM Inference (Mock)

```bash
# Test OpenVLA (will download ~14GB model on first run)
python scripts/test_vla_inference.py --vlm openvla

# Or test PaliGemma (smaller, ~6GB)
python scripts/test_vla_inference.py --vlm paligemma
```

### 3. Test Action Expert (Mock)

```bash
# Test ACT policy
python scripts/test_vla_inference.py --action act

# Or test all together
python scripts/test_vla_inference.py --vlm openvla --action act
```

## 🤖 Real Robot Deployment (SoArm 100)

### 1. Start the VLA Pipeline Server
This loads the VLM and Action Expert models on your GPUs.
```bash
# Terminal 1
cd /home/mecha/hive_mind
python services/pipeline/server.py
```

### 2. Start the ROS Adapter
Connects the VLA pipeline to your robot's ROS2 topics.
```bash
# Terminal 2
cd /home/mecha/hive_mind
python hardware/ros_adapter/ros_node.py
```

### 3. Give the Robot a Task
Use the helper script to send a natural language command.
```bash
# Terminal 3
cd /home/mecha/hive_mind
python scripts/set_instruction.py "Pick up the red cube and place it in the blue box"
```

---

## 📝 Switch Models (Zero Code Changes!)

Edit `configs/master_config.yaml`:

```yaml
vlm:
  provider: "paligemma"  # Recommended for real-time (fastest)
  # provider: "openvla"   # High accuracy (slowest)

action_expert:
  provider: "act"  # Smooth, chunked actions
  # provider: "diffusion" # Robust to noise
```

---

## 🔍 Monitoring & Performance

Check state and latency:
```bash
curl http://localhost:8000/health
```

| Metric | Target | Notes |
|--------|--------|-------|
| VLM Latency | ~100ms | Runs at 1Hz by default |
| Action Latency | ~20ms | Runs at 20Hz |
| Total E2E | <150ms | Key for stable control |

---

## ⚡ What's Implemented

✅ **VLM Providers**: OpenVLA, PaliGemma, Gemini, GR00T (placeholder)  
✅ **Action Experts**: ACT, Diffusion, Pi0  
✅ **Infrastructure**: Checkpoint manager, GPU utils, master config  
✅ **Testing**: Mock inference tests with latency measurement  
✅ **Robot Config**: SoArm 100 specification  

---

## 🎯 Next Steps

1. **Test with real camera** - Update perception to use actual images
2. **Download checkpoints** - Models auto-download from HuggingFace
3. **Run on SoArm 100** - Integrate with your ROS2 pipeline
4. **Collect data** - Use for fine-tuning action experts
5. **Fine-tune** - Train on your SoArm-specific tasks

---

**Ready to move real robots! 🦾**
