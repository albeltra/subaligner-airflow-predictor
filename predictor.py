import datetime
import gc
import json
import os
import threading
from sklearn.metrics import log_loss
import traceback
from copy import deepcopy
from typing import Tuple, List, Optional, Dict, Any, Union

import h5py
import numpy as np
from pysrt import SubRipItem, SubRipFile
from subaligner.old_predictor import Predictor as OldPredictor

from .embedder import FeatureEmbedder
from .exception import TerminalException, NoFrameRateException
from .network import Network
from .subtitle import Subtitle


class Predictor(OldPredictor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def predict_single_pass(
            self,
            video_file_path: str,
            subtitle_file_path: str,
            weights_dir: str = os.path.join(os.path.dirname(__file__), "models", "training", "weights"),
            channel: str = '0',
            audio_file_path: str = None
    ) -> Tuple[List[SubRipItem], str, Union[np.ndarray, List[float]], Optional[float]]:
        """Predict time to shift with single pass

            Arguments:
                video_file_path {string} -- The input video file path.
                audio_file_path {string} -- Optional audio file path.
                subtitle_file_path {string} -- The path to the subtitle file.
                weights_dir {string} -- The the model weights directory.

            Returns:
                tuple -- The shifted subtitles, the audio file path and the voice probabilities of the original audio.
        """

        weights_file_path = self.__get_weights_path(weights_dir)
        audio_file_path = audio_file_path if audio_file_path else None
        frame_rate = None
        try:
            subs, audio_file_path, voice_probabilities = self.__predict(
                video_file_path, subtitle_file_path, weights_file_path, audio_file_path, channel=channel
            )
            try:
                frame_rate = self.__media_helper.get_frame_rate(video_file_path)
                self.__feature_embedder.step_sample = 1 / frame_rate
                self.__on_frame_timecodes(subs)
            except NoFrameRateException:
                self.__LOGGER.warning("Cannot detect the frame rate for %s" % video_file_path)
            return subs, audio_file_path, voice_probabilities, frame_rate
        finally:
            pass

    def __predict(
            self,
            video_file_path: Optional[str],
            subtitle_file_path: Optional[str],
            weights_file_path: str,
            audio_file_path: Optional[str] = None,
            subtitles: Optional[SubRipFile] = None,
            max_shift_secs: Optional[float] = None,
            previous_gap: Optional[float] = None,
            lock: Optional[threading.RLock] = None,
            network: Optional[Network] = None,
            channel: str = '0'
    ) -> Tuple[List[SubRipItem], str, np.ndarray]:
        """Shift out-of-sync subtitle cues by sending the audio track of an video to the trained network.

        Arguments:
            video_file_path {string} -- The file path of the original video.
            subtitle_file_path {string} -- The file path of the out-of-sync subtitles.
            weights_file_path {string} -- The file path of the weights file.

        Keyword Arguments:
            audio_file_path {string} -- The file path of the original audio (default: {None}).
            subtitles {list} -- The list of SubRip files (default: {None}).
            max_shift_secs {float} -- The maximum seconds by which subtitle cues can be shifted (default: {None}).
            previous_gap {float} -- The duration between the start time of the audio segment and the start time of the subtitle segment (default: {None}).

        Returns:
            tuple -- The shifted subtitles, the audio file path and the voice probabilities of the original audio.
        """
        if network is None:
            network = self.__initialise_network(os.path.dirname(weights_file_path), self.__LOGGER)
        result: Dict[str, Any] = {}
        pred_start = datetime.datetime.now()
        if audio_file_path is not None:
            result["audio_file_path"] = audio_file_path
        elif video_file_path is not None:
            t = datetime.datetime.now()
            audio_file_path = self.__media_helper.extract_audio(
                video_file_path, True, 16000, channel=channel
            )
            self.__LOGGER.debug(
                "[{}] Audio extracted after {}".format(
                    os.getpid(), str(datetime.datetime.now() - t)
                )
            )
            result["video_file_path"] = video_file_path
        else:
            raise TerminalException("Neither audio nor video is passed in")

        subs = None 
        if subtitle_file_path is not None:
            for extension in ['.vob.srt', '.pgs.srt', '.srt.srt']:
                if os.path.exists(subtitle_file_path + extension):
                    subs = Subtitle.load(subtitle_file_path + extension).subs
                else:
                    continue
            result["subtitle_file_path"] = subtitle_file_path
        elif subtitles is not None:
            subs = subtitles
        else:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            raise TerminalException("ERROR: No subtitles passed in")

        train_data = None
        labels = None

        data_path = video_file_path + '.hdf5'
        if os.path.exists(data_path):
            with h5py.File(data_path, 'r') as f:
                train_data = np.array(f['data'])[np.newaxis, ...]
                labels = np.array(f['labels'])[np.newaxis, ...]

        try:
            if train_data is None and labels is None:
                train_data, labels = self.__feature_embedder.extract_data_and_label_from_audio(
                    audio_file_path, None, subtitles=subs
                )
                with h5py.File(data_path, 'a') as f:
                    f['data'] = train_data
                    f['labels'] = labels
        except TerminalException:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            raise

        # train_data = np.array([np.rot90(val) for val in train_data])
        # train_data = train_data - np.mean(train_data, axis=0)
        result["time_load_dataset"] = (datetime.datetime.now() - pred_start).total_seconds()
        result["X_shape"] = train_data.shape[1]

        # Load neural network
        input_shape = (train_data.shape[1], train_data.shape[2])
        self.__LOGGER.debug("[{}] input shape: {}".format(os.getpid(), input_shape))

        # Network class is not thread safe so a new graph is created for each thread
        pred_start = datetime.datetime.now()
        if lock is not None:
            with lock:
                try:
                    self.__LOGGER.info("[{}] Start predicting...".format(os.getpid()))
                    voice_probabilities = network.get_predictions(train_data, weights_file_path)[0]
                    self.__LOGGER.info("Done predicting")
                except Exception as e:
                    self.__LOGGER.error("[{}] Prediction failed: {}\n{}".format(os.getpid(), str(e), "".join(traceback.format_stack())))
                    traceback.print_tb(e.__traceback__)
                    raise TerminalException("Prediction failed") from e
                finally:
                    del train_data
                    del labels
                    gc.collect()
        else:
            try:
                self.__LOGGER.debug("[{}] Start predicting...".format(os.getpid()))
                voice_probabilities = network.get_predictions(train_data, weights_file_path)[0]
            except Exception as e:
                self.__LOGGER.error(
                    "[{}] Prediction failed: {}\n{}".format(os.getpid(), str(e), "".join(traceback.format_stack())))
                traceback.print_tb(e.__traceback__)
                raise TerminalException("Prediction failed") from e
            finally:
                del train_data
                del labels
                gc.collect()

        if len(voice_probabilities) <= 0:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            raise TerminalException(
                "ERROR: Audio is too short and no voice was detected"
            )

        result["time_predictions"] = (datetime.datetime.now() - pred_start).total_seconds()

        original_start = FeatureEmbedder.time_to_sec(subs[0].start)
        shifted_subs = deepcopy(subs)
        subs.shift(seconds=-original_start)

        self.__LOGGER.info("[{}] Aligning subtitle with video...".format(os.getpid()))

        if lock is not None:
            with lock:
                min_log_loss, min_log_loss_pos = self.get_min_log_loss_and_index(
                    voice_probabilities, subs
                )
        else:
            min_log_loss, min_log_loss_pos = self.get_min_log_loss_and_index(
                voice_probabilities, subs
            )

        pos_to_delay = min_log_loss_pos
        result["loss"] = min_log_loss

        self.__LOGGER.info("[{}] Subtitle aligned".format(os.getpid()))

        if subtitle_file_path is not None:  # for the first pass
            seconds_to_shift = (
                self.__feature_embedder.position_to_duration(pos_to_delay) - original_start
            )
        elif subtitles is not None:  # for each in second pass
            seconds_to_shift = self.__feature_embedder.position_to_duration(pos_to_delay) - previous_gap if previous_gap is not None else 0.0
        else:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            raise ValueError("ERROR: No subtitles passed in")

        if abs(seconds_to_shift) > Predictor.__MAX_SHIFT_IN_SECS:
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)
            raise TerminalException(
                "Average shift duration ({} secs) have been reached".format(
                    Predictor.__MAX_SHIFT_IN_SECS
                )
            )

        result["seconds_to_shift"] = seconds_to_shift
        result["original_start"] = original_start
        total_elapsed_time = (datetime.datetime.now() - pred_start).total_seconds()
        result["time_sync"] = total_elapsed_time
        self.__LOGGER.debug("[{}] Statistics: {}".format(os.getpid(), result))

        self.__LOGGER.debug("[{}] Total Time: {}".format(os.getpid(), total_elapsed_time))
        self.__LOGGER.debug(
            "[{}] Seconds to shift: {}".format(os.getpid(), seconds_to_shift)
        )

        # For each subtitle chunk, its end time should not be later than the end time of the audio segment
        if max_shift_secs is not None and seconds_to_shift <= max_shift_secs:
            shifted_subs.shift(seconds=seconds_to_shift)
        elif max_shift_secs is not None and seconds_to_shift > max_shift_secs:
            self.__LOGGER.warning(
                "[{}] Maximum {} seconds shift has reached".format(os.getpid(), max_shift_secs)
            )
            shifted_subs.shift(seconds=max_shift_secs)
        else:
            shifted_subs.shift(seconds=seconds_to_shift)
        self.__LOGGER.debug("[{}] Subtitle shifted".format(os.getpid()))

        modified_result = {}
        for key, value in result.items():
            modified_result['SUBALIGNER_' + key] = value

        modified_result['SUBALIGNER_Extension'] = video_file_path.split('.')[-1]
        with open("/airflow/xcom/return.json", "w") as f:
            json.dump(modified_result, f)
        return shifted_subs, audio_file_path, voice_probabilities
