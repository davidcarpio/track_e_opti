import sys
import os

# Minimal pytest mock
class PytestMock:
    class approx:
        def __init__(self, expected, rel=None, abs=None, nan_ok=False):
            self.expected = expected
            self.rel = rel
            self.abs = abs

        def __eq__(self, actual):
            if self.abs is not None:
                return abs(self.expected - actual) <= self.abs
            if self.rel is not None:
                return abs(self.expected - actual) <= abs(self.expected) * self.rel
            # default to 1e-6
            return abs(self.expected - actual) <= 1e-6
            
    def fixture(self, func):
        return func
        
sys.modules['pytest'] = PytestMock()

# Now import the test file
import tests.test_per_wheel_dynamics as t

uc_vehicle = t.uc_vehicle()
proto_vehicle = t.proto_vehicle()

def run_tests_in_class(cls, *fixtures):
    inst = cls()
    for name in dir(inst):
        if name.startswith('test_'):
            method = getattr(inst, name)
            import inspect
            sig = inspect.signature(method)
            kwargs = {}
            for param in sig.parameters:
                if param == 'uc_vehicle':
                    kwargs['uc_vehicle'] = uc_vehicle
                elif param == 'proto_vehicle':
                    kwargs['proto_vehicle'] = proto_vehicle
            try:
                method(**kwargs)
                print(f"PASS: {cls.__name__}.{name}")
            except Exception as e:
                print(f"FAIL: {cls.__name__}.{name} - {e}")
                raise e

print("Running Tests...")
run_tests_in_class(t.TestAxleNormalForces)
run_tests_in_class(t.TestMaxDriveForce)
run_tests_in_class(t.TestMaxBrakingForce)
run_tests_in_class(t.TestRolloverLimit)
run_tests_in_class(t.TestMaxCorneringVelocity)
run_tests_in_class(t.TestConfigProperties)
run_tests_in_class(t.TestPresets)
run_tests_in_class(t.TestBackwardCompatibility)
print("All Tests Passed!")
