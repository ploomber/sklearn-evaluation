"""
Test configuration file, ignore
"""
import matplotlib.pyplot as plt

# mock plt.show() so plots do not appear when running the tests
plt.show = lambda: None


def pytest_sessionfinish(session, exitstatus):
    """
    This avoids travis breaking due to the 'no tests found' 5 exit status
    """
    if exitstatus == 5:
        session.exitstatus = 0
