from jax._src import linear_util

from src.main import main

original_store = linear_util.EqualStore.store


def _new_store(self: linear_util.EqualStore, val):
    # Necessary since 0.4.10
    # Alternative: Manually unroll the for scan in src.model.main.body_ctx to a for loop - increasing compile times
    try:
        original_store(self, val)
    except linear_util.StoreException:
        if str(self._store._val) == str(val):
            return
        raise


linear_util.EqualStore.store = _new_store

if __name__ == '__main__':
    main()
