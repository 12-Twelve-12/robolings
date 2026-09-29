import loader


def pytest_report_header(config):
    return f"robolings target: {loader.TARGET_DIR}"
