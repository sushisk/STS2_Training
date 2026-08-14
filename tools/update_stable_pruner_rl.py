import sys
import update_stable_pruner_rl_v3_setup
import update_stable_pruner_rl_v2_impl as impl

if __name__ == "__main__":
    raise SystemExit(impl.main())
sys.modules[__name__] = impl
