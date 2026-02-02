import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from hardware.ros_adapter.safety_validator import SafetyValidator
import yaml

def test_joint_limits():
    """Test joint limit enforcement"""
    print("\n=== Test 1: Joint Limits ===")
    
    config = {
        "joints": {
            "names": ["J1", "J2", "J3"],
            "limits": {
                "min": [-1.0, -1.0, -1.0],
                "max": [1.0, 1.0, 1.0]
            },
            "velocity_limits": [0.5, 0.5, 0.5]
        },
        "safety": {"enable_limits": True}
    }
    
    validator = SafetyValidator(config)
    
    # Test 1a: Valid command
    is_safe, safe_pos, reason = validator.validate_command([0.5, 0.5, 0.5])
    assert is_safe and reason == "safe", f"Valid command rejected: {reason}"
    print("✓ Valid command accepted")
    
    # Test 1b: Out of bounds (should clamp)
    is_safe, safe_pos, reason = validator.validate_command([2.0, 0.0, -2.0])
    assert is_safe and reason != "safe", "Out-of-bounds not detected"
    assert safe_pos == [1.0, 0.0, -1.0], f"Clamping failed: {safe_pos}"
    print(f"✓ Out-of-bounds clamped: {reason}")
    
    # Test 1c: Statistics
    stats = validator.get_statistics()
    assert stats["joint_limits"] == 1, "Statistics not updated"
    print(f"✓ Statistics tracked: {stats}")

def test_emergency_stop():
    """Test emergency stop functionality"""
    print("\n=== Test 2: Emergency Stop ===")
    
    config = {
        "joints": {
            "names": ["J1"],
            "limits": {"min": [-1.0], "max": [1.0]},
            "velocity_limits": [0.5]
        },
        "safety": {"enable_limits": True}
    }
    
    validator = SafetyValidator(config)
    
    # Test 2a: Normal operation
    is_safe, _, reason = validator.validate_command([0.5])
    assert is_safe and reason == "safe"
    print("✓ Normal operation OK")
    
    # Test 2b: Activate emergency stop
    validator.set_emergency_stop(True)
    is_safe, _, reason = validator.validate_command([0.5])
    assert not is_safe and reason == "emergency_stop_active"
    print("✓ Emergency stop blocks commands")
    
    # Test 2c: Release emergency stop
    validator.set_emergency_stop(False)
    is_safe, _, reason = validator.validate_command([0.5])
    assert is_safe and reason == "safe"
    print("✓ Emergency stop released")

def test_velocity_limits():
    """Test velocity limit checking"""
    print("\n=== Test 3: Velocity Limits ===")
    
    config = {
        "joints": {
            "names": ["J1", "J2"],
            "limits": {"min": [-1.0, -1.0], "max": [1.0, 1.0]},
            "velocity_limits": [0.5, 0.5]
        },
        "safety": {"enable_limits": True}
    }
    
    validator = SafetyValidator(config)
    
    # Test with high velocity
    is_safe, _, reason = validator.validate_command([0.5, 0.5], velocities=[1.0, 0.2])
    assert "velocity" in reason.lower(), f"Velocity violation not detected: {reason}"
    print(f"✓ Velocity limit violation detected: {reason}")

def test_collision_heuristics():
    """Test basic collision checking"""
    print("\n=== Test 4: Collision Heuristics ===")
    
    config = {
        "joints": {
            "names": ["Base", "Shoulder", "Elbow"],
            "limits": {"min": [-3.14, -3.14, -3.14], "max": [3.14, 3.14, 3.14]},
            "velocity_limits": [1.0, 1.0, 1.0]
        },
        "safety": {
            "enable_limits": True,
            "collision_checks": {
                "enable_self_collision": True,
                "min_elbow_angle": 0.1
            }
        }
    }
    
    validator = SafetyValidator(config)
    
    # Test with extreme elbow bend
    is_safe, _, reason = validator.validate_command([0.0, 0.0, 0.05])
    assert "collision" in reason.lower(), f"Collision not detected: {reason}"
    print(f"✓ Self-collision detected: {reason}")

def run_all_tests():
    """Run complete safety test suite"""
    print("=" * 50)
    print("SAFETY VALIDATOR TEST SUITE")
    print("=" * 50)
    
    try:
        test_joint_limits()
        test_emergency_stop()
        test_velocity_limits()
        test_collision_heuristics()
        
        print("\n" + "=" * 50)
        print("ALL TESTS PASSED ✓")
        print("=" * 50)
        return True
        
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
