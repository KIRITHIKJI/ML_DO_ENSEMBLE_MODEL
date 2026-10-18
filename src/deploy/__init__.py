"""
STEP 12 - the deployment layer.

Everything in this package sits on top of the frozen system rather than
inside it. It loads config.yaml for the science and config_deploy.yaml for
the deployment, reuses the project's own download / clean / feature /
sequence / metric code, and never trains, tunes or rescores anything that
the paper reports.
"""
