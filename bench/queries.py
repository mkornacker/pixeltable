import pixeltable as pxt

t = pxt.get_table('bench.lookup')


@pxt.query
def q1(i: int):
    return t.where(t.id == i).select(doubled=t.doubled)
