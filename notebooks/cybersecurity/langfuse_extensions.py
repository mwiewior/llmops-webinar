import threading
import types
from datetime import datetime, timedelta

import dspy
import pandas as pd
from dspy import Evaluate
import tqdm
from dspy.evaluate.evaluate import merge_dicts, truncate_cell, stylize_metric_name, display_dataframe, display, HTML

from langfuse.model import ModelUsage
from dsp.trackers.langfuse_tracker import LangfuseTracker


class DSPyLangfuseTracker(LangfuseTracker):
    def call(self, i, o, name=None, session=None, score=None, ground_truth=None, pred=None, **kwargs):
        trace_client = self.langfuse.trace(session_id=session, input=i, output=o, name=name, metadata=kwargs)
        # Unpack args if they are being used to pass i, o, etc.
        model = o['model'] if 'model' in o else None
        output = o['choices'][0]['message']['content'] if 'choices' in o else None
        prompt_tokens = o['usage']['prompt_tokens'] if 'usage' in o else None
        completion_tokens = o['usage']['completion_tokens'] if 'usage' in o else None
        date_epoch = o['created'] if 'created' in o else None
        total_duration = o['additional_kwargs']['total_duration'] if 'additional_kwargs' in o else 0
        # Convert epoch time to datetime object
        start_time = datetime.fromtimestamp(date_epoch)
        model_parameters = kwargs
        usage = ModelUsage(
            input=prompt_tokens,
            output=completion_tokens
        )
        generation_client = self.langfuse.generation(
            trace_id=trace_client.id,
            start_time=start_time,
            end_time=(start_time + timedelta(milliseconds=total_duration / 1000000)),
            input=i, output=output, name=name, model=model, usage=usage, model_parameters=model_parameters
        )
        self.langfuse.score(
            trace_id=trace_client.id,
            observation_id=generation_client.id,  # optional
            name="model",
            value=model
        )
        self.langfuse.score(
            trace_id=trace_client.id,
            observation_id=generation_client.id,  # optional
            name="accuracy",
            value=score,
            data_type="NUMERIC",  # optional, inferred if not provided
        )
        if ground_truth:
            self.langfuse.score(
                trace_id=trace_client.id,
                observation_id=generation_client.id,  # optional
                name="ground_truth",
                value=ground_truth
            )
        if pred:
            self.langfuse.score(
                trace_id=trace_client.id,
                observation_id=generation_client.id,  # optional
                name="prediction",
                value=pred
            )


class EvaluateWithLangfuse(Evaluate):

    def __init__(self, run_id, **kwargs):
        self.run_id = run_id
        super().__init__(**kwargs)

    def __call__(
            self,
            program,
            metric=None,
            devset=None,
            num_threads=None,
            display_progress=None,
            display_table=None,
            return_all_scores=None,
            return_outputs=None,
    ):
        metric = metric if metric is not None else self.metric
        devset = devset if devset is not None else self.devset
        num_threads = num_threads if num_threads is not None else self.num_threads
        display_progress = display_progress if display_progress is not None else self.display_progress
        display_table = display_table if display_table is not None else self.display_table
        return_all_scores = return_all_scores if return_all_scores is not None else self.return_all_scores
        return_outputs = return_outputs if return_outputs is not None else self.return_outputs
        results = []

        def wrapped_program(example_idx, example):
            # NOTE: TODO: Won't work if threads create threads!
            thread_stacks = dspy.settings.stack_by_thread
            creating_new_thread = threading.get_ident() not in thread_stacks
            if creating_new_thread:
                thread_stacks[threading.get_ident()] = list(dspy.settings.main_stack)

            try:
                langfuse = DSPyLangfuseTracker(
                    debug=False
                )
                prediction = program(**example.inputs())
                score = metric(
                    example,
                    prediction,
                )  # FIXME: TODO: What's the right order? Maybe force name-based kwargs!
                model = program.lm.model_name if program.lm.provider=="ollama" else program.lm.kwargs['model']
                program.lm.tracker_call(tracker=langfuse,
                                        session=self.run_id,
                                        name=f"Evaluation-{model}-{example_idx}",
                                        score=1.0 if score else 0.0,
                                        ground_truth=example.output.label,
                                        pred=prediction.output.label)
                # increment assert and suggest failures to program's attributes
                if hasattr(program, "_assert_failures"):
                    program._assert_failures += dspy.settings.get("assert_failures")
                if hasattr(program, "_suggest_failures"):
                    program._suggest_failures += dspy.settings.get("suggest_failures")

                return example_idx, example, prediction, score
            except Exception as e:
                with self.error_lock:
                    self.error_count += 1
                    current_error_count = self.error_count
                if current_error_count >= self.max_errors:
                    raise e

                if self.provide_traceback:
                    dspy.logger.error(
                        f"Error for example in dev set: \t\t {e}\n\twith inputs:\n\t\t{example.inputs()}\n\nStack trace:\n\t{traceback.format_exc()}"
                    )
                else:
                    dspy.logger.error(
                        f"Error for example in dev set: \t\t {e}. Set `provide_traceback=True` to see the stack trace."
                    )

                return example_idx, example, {}, 0.0
            finally:
                if creating_new_thread:
                    del thread_stacks[threading.get_ident()]

        devset = list(enumerate(devset))
        tqdm.tqdm._instances.clear()

        if num_threads == 1:
            reordered_devset, ncorrect, ntotal = self._execute_single_thread(wrapped_program, devset, display_progress)
        else:
            reordered_devset, ncorrect, ntotal = self._execute_multi_thread(
                wrapped_program,
                devset,
                num_threads,
                display_progress,
            )

        dspy.logger.info(f"Average Metric: {ncorrect} / {ntotal} ({round(100 * ncorrect / ntotal, 1)}%)")

        predicted_devset = sorted(reordered_devset)

        if return_outputs:  # Handle the return_outputs logic
            results = [(example, prediction, score) for _, example, prediction, score in predicted_devset]

        data = [
            merge_dicts(example, prediction) | {"correct": score} for _, example, prediction, score in predicted_devset
        ]

        result_df = pd.DataFrame(data)

        # Truncate every cell in the DataFrame (DataFrame.applymap was renamed to DataFrame.map in Pandas 2.1.0)
        result_df = result_df.map(truncate_cell) if hasattr(result_df, "map") else result_df.applymap(truncate_cell)

        # Rename the 'correct' column to the name of the metric object
        metric_name = metric.__name__ if isinstance(metric, types.FunctionType) else metric.__class__.__name__
        result_df = result_df.rename(columns={"correct": metric_name})

        if display_table:
            if isinstance(display_table, bool):
                df_to_display = result_df.copy()
                truncated_rows = 0
            else:
                df_to_display = result_df.head(display_table).copy()
                truncated_rows = len(result_df) - display_table

            df_to_display = stylize_metric_name(df_to_display, metric_name)

            display_dataframe(df_to_display)

            if truncated_rows > 0:
                # Simplified message about the truncated rows
                message = f"""
                <div style='
                    text-align: center;
                    font-size: 16px;
                    font-weight: bold;
                    color: #555;
                    margin: 10px 0;'>
                    ... {truncated_rows} more rows not displayed ...
                </div>
                """
                display(HTML(message))

        if return_all_scores and return_outputs:
            return round(100 * ncorrect / ntotal, 2), results, [score for *_, score in predicted_devset]
        if return_all_scores:
            return round(100 * ncorrect / ntotal, 2), [score for *_, score in predicted_devset]
        if return_outputs:
            return round(100 * ncorrect / ntotal, 2), results

        return round(100 * ncorrect / ntotal, 2)
