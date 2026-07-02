"""Auto-registration entry used by the install-time .pth hook.

A `.pth` file placed in site-packages with the line::

    import dataflex_verl.autoload

runs at EVERY interpreter startup in that environment — including the fresh
Python processes Ray spawns for its actors/workers. This is what makes
`config.trainer.v1.trainer_mode=dataflex_sync` resolve inside verl's Ray actors
without the user importing anything manually.

Everything is guarded: if verl isn't importable yet (or at all), we silently skip
so we never break unrelated Python invocations in the same env.
"""

import os


def _autoload():
    if os.environ.get("DATAFLEX_VERL_DISABLE_AUTOLOAD", "0") == "1":
        return
    try:
        import dataflex_verl

        dataflex_verl.register_all()
    except Exception:
        # verl not ready / not installed / partial import — never hard-fail startup
        pass


_autoload()
