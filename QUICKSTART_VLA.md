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

---

## 📝 Switch Models (Zero Code Changes!)

Edit `configs/master_config.yaml`:

```yaml
vlm:
  provider: "openvla"  # or: paligemma, gemini, simple_mlp

action_expert:
  provider: "act"  # or: diffusion, pi0, pi0_fast, scripted
```

---

## 🤖 Use in Your Code

```python
from services.task.model import TaskModelFactory
from services.action.factory import ActionExpertFactory

# Load models from config
vlm = TaskModelFactory.create()
action_expert = ActionExpertFactory.create()

# Run inference
task_latent = vlm.infer(perception, instruction)
action_chunk = action_expert.act(task_latent, perception, robot_state)
```

---

## 📚 Full Documentation

- **Walkthrough**: [walkthrough.md](file:///home/mecha/.gemini/antigravity/brain/bf40018a-ff4d-415c-aa3c-93a64be831b6/walkthrough.md)
- **Architecture**: [architecture_overview.md](file:///home/mecha/.gemini/antigravity/brain/bf40018a-ff4d-415c-aa3c-93a64be831b6/architecture_overview.md)
- **Implementation Plan**: [implementation_plan.md](file:///home/mecha/.gemini/antigravity/brain/bf40018a-ff4d-415c-aa3c-93a64be831b6/implementation_plan.md)

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
