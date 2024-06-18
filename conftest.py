from pymicro import dowload_datadir

def pytest_sessionstart(session):
    print("sessionstart", session)
    dowload_datadir()
