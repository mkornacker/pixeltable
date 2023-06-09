import pytest

from pixeltable.function import Function, FunctionRegistry
from pixeltable.type_system import IntType, FloatType
from pixeltable import catalog
import pixeltable as pt
from pixeltable import exceptions as exc


def dummy_fn(i: int) -> int:
    return i

class TestFunction:
    @pt.function(return_type=IntType(), param_types=[IntType()])
    def func(x: int) -> int:
        return x + 1

    class Aggregator:
        def __init__(self):
            self.sum = 0
        @classmethod
        def make_aggregator(cls) -> 'Aggregator':
            return cls()
        def update(self, val) -> None:
            if val is not None:
                self.sum += val
        def value(self):
            return self.sum
    agg = Function.make_aggregate_function(
        IntType(), [IntType()], Aggregator.make_aggregator, Aggregator.update, Aggregator.value)

    def test_serialize_anonymous(self, init_env) -> None:
        d = self.func.as_dict()
        FunctionRegistry.get().clear_cache()
        deserialized = Function.from_dict(d)
        assert deserialized.eval_fn(1) == 2

    def test_create(self, test_db: catalog.Db) -> None:
        db = test_db
        db.create_function('test_fn', self.func)
        assert self.func.md.fqn == 'test.test_fn'
        FunctionRegistry.get().clear_cache()
        cl = pt.Client()
        _ = cl.list_functions()
        db2 = cl.get_db('test')
        fn2 = db2.get_function('test_fn')
        assert fn2.md.fqn == 'test.test_fn'
        assert fn2.eval_fn(1) == 2

        with pytest.raises(exc.Error):
            db.create_function('test_fn', self.func)
        with pytest.raises(exc.Error):
            db.create_function('dir1.test_fn', self.func)
        with pytest.raises(exc.Error):
            library_fn = Function.make_library_function(IntType(), [IntType()], __name__, 'dummy_fn')
            db.create_function('library_fn', library_fn)

    def test_update(self, test_db: catalog.Db, test_tbl: catalog.Table) -> None:
        db = test_db
        t = test_tbl
        db.create_function('test_fn', self.func)
        res1 = t[self.func(t.c2)].show(0).to_pandas()

        # load function from db and make sure it computes the same thing as before
        FunctionRegistry.get().clear_cache()
        cl = pt.Client()
        db = cl.get_db('test')
        fn = db.get_function('test_fn')
        res2 = t[fn(t.c2)].show(0).to_pandas()
        assert res1.col_0.equals(res2.col_0)
        fn.eval_fn = lambda x: x + 2
        db.update_function('test_fn', fn)
        assert self.func.md.fqn == fn.md.fqn  # fqn doesn't change

        FunctionRegistry.get().clear_cache()
        cl = pt.Client()
        db = cl.get_db('test')
        fn = db.get_function('test_fn')
        assert self.func.md.fqn == fn.md.fqn  # fqn doesn't change
        res3 = t[fn(t.c2)].show(0).to_pandas()
        assert (res2.col_0 + 1).equals(res3.col_0)

        # signature changes
        with pytest.raises(exc.Error):
            db.update_function('test_fn', Function.make_function(FloatType(), [IntType()], fn.eval_fn))
        with pytest.raises(exc.Error):
            db.update_function('test_fn', Function.make_function(IntType(), [FloatType()], fn.eval_fn))
        with pytest.raises(exc.Error):
            db.update_function('test_fn', self.agg)

    def test_rename(self, test_db: catalog.Db) -> None:
        db = test_db
        db.create_function('test_fn', self.func)

        FunctionRegistry.get().clear_cache()
        cl = pt.Client()
        db2 = cl.get_db('test')
        with pytest.raises(exc.Error):
            db2.rename_function('test_fn2', 'test_fn')
        db2.rename_function('test_fn', 'test_fn2')
        func = db2.get_function('test_fn2')
        assert func.eval_fn(1) == 2
        assert func.md.fqn == 'test.test_fn2'

        with pytest.raises(exc.Error):
            _ = db2.get_function('test_fn')

        # move function between directories
        db2.create_dir('functions')
        db2.create_dir('functions2')
        db2.create_function('functions.func1', self.func)
        with pytest.raises(exc.Error):
            db2.rename_function('functions2.func1', 'functions.func1')
        db2.rename_function('functions.func1', 'functions2.func1')
        func = db2.get_function('functions2.func1')
        assert func.md.fqn == 'test.functions2.func1'


        FunctionRegistry.get().clear_cache()
        cl = pt.Client()
        db3 = cl.get_db('test')
        func = db3.get_function('functions2.func1')
        assert func.eval_fn(1) == 2
        assert func.md.fqn == 'test.functions2.func1'
        with pytest.raises(exc.Error):
            _ = db3.get_function('functions.func1')

    def test_drop(self, test_db: catalog.Db) -> None:
        db = test_db
        db.create_function('test_fn', self.func)
        FunctionRegistry.get().clear_cache()
        cl = pt.Client()
        db2 = cl.get_db('test')
        db2.drop_function('test_fn')

        with pytest.raises(exc.Error):
            _ = db2.get_function('test_fn')

    def test_list(self, test_db: catalog.Db) -> None:
        _ = FunctionRegistry.get().list_functions()
        print(_)
