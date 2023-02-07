import pytest
import PIL
import ffmpeg

import pixeltable as pt
from pixeltable.type_system import VideoType, IntType, ImageType
from pixeltable.tests.utils import get_video_files
from pixeltable import catalog
from pixeltable import utils
from pixeltable import exceptions as exc


class TestVideo:
    def test_basic(self, test_db: catalog.Db) -> None:
        video_filepaths = get_video_files()
        db = test_db
        cols = [
            catalog.Column('video', VideoType(), nullable=False),
            catalog.Column('frame', ImageType(), nullable=False),
            catalog.Column('frame_idx', IntType(), nullable=False),
        ]
        # extract frames at fps=1
        tbl = db.create_table(
            'test', cols, extract_frames_from='video', extracted_frame_col='frame',
            extracted_frame_idx_col='frame_idx', extracted_fps=1)
        tbl.insert_rows([[p] for p in video_filepaths], columns=['video'])
        tbl_count = tbl.count()
        assert utils.extracted_frame_count(tbl_id=tbl.id) == tbl.count()
        _ = tbl[tbl.frame_idx, tbl.frame, tbl.frame.rotate(90)].show(0)

        # the same, but expressed as a filter
        tbl2 = db.create_table(
            'test2', cols, extract_frames_from='video', extracted_frame_col='frame',
            extracted_frame_idx_col='frame_idx', extracted_fps=0,
            ffmpeg_filter={'select': 'isnan(prev_selected_t)+gte(t-prev_selected_t, 1)'})
        tbl2.insert_rows([[p] for p in video_filepaths], columns=['video'])
        tbl2_count = tbl2.count()
        assert utils.extracted_frame_count(tbl_id=tbl2.id) == tbl2.count()
        # for some reason there's one extra frame in tbl2
        assert tbl.count() == tbl2.count() - 1

        # missing 'columns' arg
        with pytest.raises(exc.Error):
            tbl.insert_rows([[p] for p in video_filepaths])

        # column values mismatch in rows
        with pytest.raises(exc.Error):
            tbl.insert_rows([[1, 2], [3]], columns=['video'])

        # column values mismatch in rows
        with pytest.raises(exc.Error):
            tbl.insert_rows([[1, 2]], columns=['video'])

        # revert() also removes extracted frames
        tbl.revert()
        assert utils.extracted_frame_count(tbl_id=tbl.id) == tbl.count()

    def test_make_video(self, test_db: catalog.Db) -> None:
        video_filepaths = get_video_files()
        db = test_db
        cols = [
            catalog.Column('video', VideoType(), nullable=False),
            catalog.Column('frame', ImageType(), nullable=False),
            catalog.Column('frame_idx', IntType(), nullable=False),
        ]
        t = db.create_table(
            'test', cols, extract_frames_from = 'video', extracted_frame_col = 'frame',
            extracted_frame_idx_col = 'frame_idx', extracted_fps = 1)
        t.insert_rows([[p] for p in video_filepaths], columns=['video'])
        _ = t[pt.make_video(t.frame_idx, t.frame)].group_by(t.video).show()
        print(_)

        with pytest.raises(exc.Error):
            # make_video() doesn't allow windows
            _ = t[pt.make_video(t.frame_idx, t.frame, group_by=t.video)].show()
        with pytest.raises(exc.Error):
            # make_video() doesn't allow windows
            _ = t[pt.make_video(t.frame, order_by=t.frame_idx)].show()

        class WindowAgg:
            def __init__(self):
                self.sum = 0
            @classmethod
            def make_aggregator(cls) -> 'WindowAgg':
                return cls()
            def update(self, frame: PIL.Image.Image) -> None:
                self.sum += 1
            def value(self) -> str:
                return self.sum

        agg_fn = pt.make_aggregate_function(
            IntType(), [ImageType()],
            init_fn=WindowAgg.make_aggregator,
            update_fn=WindowAgg.update,
            value_fn=WindowAgg.value,
            requires_order_by=True, allows_std_agg=False, allows_window=True)
        # make sure it works
        _ = t[agg_fn(t.frame_idx, t.frame, group_by=t.video)].show()
        db.create_function('agg_fn', agg_fn)
        # reload from store
        cl = pt.Client()
        db = cl.get_db('test')
        agg_fn = db.get_function('agg_fn')
        t = db.get_table('test')
        _ = t[agg_fn(t.frame_idx, t.frame, group_by=t.video)].show()
        print(_)

    def test_extraction(self, test_db: catalog.Db) -> None:
        video_filepaths = get_video_files()
        _ = utils.video.extract_frames(video_filepaths[2], '/home/marcel/tmp-frames/a', fps=0, ffmpeg_filter={'select': 'gte(n,650)'})
        print(1)
