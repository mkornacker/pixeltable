from typing import AnyStr

import mcp_server.config

import pixeltable as pxt
import pixeltable.functions as pxtf
import pixeltable.iterators as pxti

settings = mcp_server.config.get_settings()


#########################################
# Models / Tables
#########################################


class Videos(pxt.Model):
    __tablename__ = 'videos'

    video: pxt.Video
    audio_extract = pxtf.video.extract_audio(M.video)


class AudioChunks(pxt.ViewModel):
    __tablename__ = 'audio_chunks'
    __base__ = pxti.AudioSplitter(
        audio=Videos.audio_extract,
        chunk_duration_sec=settings.CHUNK_DURATION,
        overlap_sec=settings.AUDIO_OVERLAP_SECONDS,
    )

    transcription = pxtf.whisper.transcribe(audio=M.audio_chunk, model='base.en')
    chunk_text = M.transcript.text

    class Meta:
        indexes = [
            pxt.EmbeddingIndex(
                M.chunk_text, string_embed=pxtf.huggingface.sentence_transformer.using(model_id=settings.MODEL_ID)
            )
        ]


@pxtf.udf
async def get_caption(prompt: str, im_str: str) -> str:
    chat_completion = pxtf.groq.chat_completions(
        messages=[
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': f'data:image/jpeg;base64,{im_str}'}},
                ],
            }
        ],
        model=settings.GROQ_VLM_MODEL,
    )
    return chat_completion.choices[0].message.content


class VideoChunks(pxt.ViewModel):
    __tablename__ = 'video_chunks'
    __base__ = pxti.FrameIterator(video=Videos.video, fps=settings.VIDEO_FPS)

    im_caption = get_caption(settings.CAPTION_MODEL_PROMPT, im_str=M.frame.b64_encode(image_format='jpeg'))

    class Meta:
        indexes = [
            pxt.EmbeddingIndex(
                M.frame, image_embed=pxtf.huggingface.clip.using(model_id=settings.IMAGE_SIMILARITY_EMBD_MODEL)
            ),
            pxt.EmbeddingIndex(
                M.im_caption, string_embed=pxtf.huggingface.clip.using(model_id=settings.CAPTION_SIMILARITY_EMBD_MODEL)
            ),
        ]


#########################################
# Search Engine
#########################################


class VideoSearchEngine:
    video_name: str

    def __init__(self, video_name: str):
        self.video_name = video_name

    def search_by_speech(self, query: str, top_k: int) -> list[dict[str, Any]]:
        t = AudioChunks
        sim = t.chunk_text.similarity(query)
        results = (
            t.where(t.video == self.video_name)
            .select(t.pos, t.start_time_secs, t.end_time_secs, similarity=sim)
            .order_by(sim, asc=False)
            .limit(top_k)
        )
        return [
            {
                'start_time': float(entry['start_time_sec']),
                'end_time': float(entry['end_time_sec']),
                'similarity': float(entry['similarity']),
            }
            for entry in results.collect()
        ]


#########################################
# Tool implememtations
#########################################


def process_video(video_path: str) -> None:
    Videos.get_table().insert({'video': video_path})


def get_video_clip_from_user_query(video_path: str, user_query: str) -> dict[str, str]:
    """Get a video clip based on the user query using speech and caption similarity.

    Args:
        video_path (str): The path to the video file.
        user_query (str): The user query to search for.

    Returns:
        Dict[str, str]: Dictionary containing:
            filename (str): Path to the extracted video clip.
    """
    search_engine = VideoSearchEngine(video_path)

    speech_clips = search_engine.search_by_speech(user_query, settings.VIDEO_CLIP_SPEECH_SEARCH_TOP_K)
    caption_clips = search_engine.search_by_caption(user_query, settings.VIDEO_CLIP_CAPTION_SEARCH_TOP_K)

    speech_sim = speech_clips[0]['similarity'] if speech_clips else 0
    caption_sim = caption_clips[0]['similarity'] if caption_clips else 0

    video_clip_info = speech_clips[0] if speech_sim > caption_sim else caption_clips[0]

    video_clip = extract_video_clip(
        video_path=video_path,
        start_time=video_clip_info['start_time'],
        end_time=video_clip_info['end_time'],
        output_path=f'./videos/{str(uuid4())}.mp4',
    )

    return {'filename': video_clip.filename}


#########################################
# Server setup
#########################################


def add_mcp_tools(mcp: FastMCP) -> None:
    mcp.add_tool(name='process_video', description='...', fn=process_video, tags=['video', 'process'])
    mcp.add_tool(
        name='get_video_clip_from_user_query',
        description='Use this tool to get a video clip from a video file based on a user query or question.',
        fn=get_video_clip_from_user_query,
        tags={'video', 'clip', 'query', 'question'},
    )
