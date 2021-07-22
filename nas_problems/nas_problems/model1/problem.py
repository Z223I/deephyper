"""Problem description."""

from deephyper.problem import NaProblem
from nas_problems.nas_problems.model1.load_data import load_data
from nas_problems.nas_problems.model1.search_space import create_search_space
from deephyper.nas.preprocessing import minmaxstdscaler

Problem = NaProblem(seed=2019)

Problem.load_data( load_data )

Problem.preprocessing(minmaxstdscaler)

Problem.search_space(create_search_space, num_layers=3)

"""
Traceback (most recent call last):
  File "/lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/deephyper/evaluator/_ray_evaluator.py", line 30, in _poll
    self._result = ray.get(id_done[0])
  File "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/ray/_private/client_mode_hook.py", line 62, in wrapper
    return func(*args, **kwargs)
  File "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/ray/worker.py", line 1495, in get
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(ValueError): ray::run() (pid=2180163, ip=10.230.2.200)
  File "python/ray/_raylet.pyx", line 501, in ray._raylet.execute_task
  File "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/ray/util/tracing/tracing_helper.py", line 330, in _function_with_tracing
    return function(*args, **kwargs)
  File "/lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/deephyper/nas/run/alpha.py", line 114, in run
    history = trainer.train(with_pred=with_pred, last_only=last_only)
  File "/lus/theta-fs0/projects/datascience/wilsonb/theta/deephyper/deephyper/nas/trainer/train_valid.py", line 451, in train
    history = self.model.fit(
  File "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py", line 1100, in fit
    tmp_logs = self.train_function(iterator)
  File "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py", line 828, in __call__
    result = self._call(*args, **kwds)
  File "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py", line 871, in _call
    self._initialize(args, kwds, add_initializers_to=initializers)
  File "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py", line 725, in _initialize
    self._stateful_fn._get_concrete_function_internal_garbage_collected(  # pylint: disable=protected-access
  File "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/eager/function.py", line 2969, in _get_concrete_function_internal_garbage_collected
    graph_function, _ = self._maybe_define_function(args, kwargs)
  File "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/eager/function.py", line 3361, in _maybe_define_function
    graph_function = self._create_graph_function(args, kwargs)
  File "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/eager/function.py", line 3196, in _create_graph_function
    func_graph_module.func_graph_from_py_func(
  File "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/framework/func_graph.py", line 990, in func_graph_from_py_func
    func_outputs = python_func(*func_args, **func_kwargs)
  File "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/eager/def_function.py", line 634, in wrapped_fn
    out = weak_wrapped_fn().__wrapped__(*args, **kwds)
  File "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/framework/func_graph.py", line 977, in wrapper
    raise e.ag_error_metadata.to_exception(e)
ValueError: in user code:




    /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:805 train_function  *
        return step_function(self, iterator)
    /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:795 step_function  **
        outputs = model.distribute_strategy.run(run_step, args=(data,))
    /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/distribute/distribute_lib.py:1259 run
        return self._extended.call_for_each_replica(fn, args=args, kwargs=kwargs)
    /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/distribute/distribute_lib.py:2730 call_for_each_replica
        return self._call_for_each_replica(fn, args, kwargs)
    /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/distribute/distribute_lib.py:3417 _call_for_each_replica
        return fn(*args, **kwargs)
    /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:788 run_step  **
        outputs = model.train_step(data)
    /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/keras/engine/training.py:758 train_step
        self.compiled_metrics.update_state(y, y_pred, sample_weight)
    /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/keras/engine/compile_utils.py:387 update_state
        self.build(y_pred, y_true)
    /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/keras/engine/compile_utils.py:317 build
        self._metrics = nest.map_structure_up_to(y_pred, self._get_metric_objects,
    /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/util/nest.py:1159 map_structure_up_to
        return map_structure_with_tuple_paths_up_to(
    /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/util/nest.py:1257 map_structure_with_tuple_paths_up_to
        results = [
    /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/util/nest.py:1258 <listcomp>
        func(*args, **kwargs) for args in zip(flat_path_gen, *flat_value_gen)
    /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/util/nest.py:1161 <lambda>
        lambda _, *values: func(*values),  # Discards the path arg.
    /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/keras/engine/compile_utils.py:418 _get_metric_objects
        return [self._get_metric_object(m, y_t, y_p) for m in metrics]
    /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/keras/engine/compile_utils.py:418 <listcomp>
        return [self._get_metric_object(m, y_t, y_p) for m in metrics]
    /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/keras/engine/compile_utils.py:437 _get_metric_object
        metric_obj = metrics_mod.get(metric)
    /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/keras/metrics.py:3490 get
        return deserialize(str(identifier))
    /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/keras/metrics.py:3446 deserialize
        return deserialize_keras_object(
    /lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/lib/python3.8/site-packages/tensorflow/python/keras/utils/generic_utils.py:377 deserialize_keras_object
        raise ValueError(

    ValueError: Unknown metric function: val_loss
"""

Problem.hyperparameters(
    batch_size=32,
    learning_rate=0.01,
    optimizer='adam',
    num_epochs=20,
    callbacks=dict(
        ModelCheckpoint=dict(
                        monitor="val_loss",
                        mode="min",
                        save_best_only=True,
                        verbose=0,
                        filepath="model.h5",
                        save_weights_only=False,
                    ),
        EarlyStopping=dict(
            monitor='val_loss', # 'val_loss', 'val_r2' or 'val_acc' ?
            mode='min',
            verbose=0,
            patience=5
        )
    )
)


Problem.loss('binary_crossentropy') # 'mse', 'binary_crossentropy' or 'categorical_crossentropy' ?

# Metrics: accuracy and validation loss.
Problem.metrics(['acc', 'val_loss']) # 'r2', 'acc', 'val_loss'

Problem.objective('val_acc__last') # 'val_r2__last', 'val_acc__last'


# Get model.
if __name__ == '__main__':
    arch_seq = [0.8323024001314224, 0.7889290619064494, 0.9385678954207153, 0.16059637997392429,
    0.6488539744120456, 0.8325765404421139, 0.9888735139157468, 0.9322143923769549,
    0.7933123406870071]
    model = Problem.get_keras_model(arch_seq)

    print('Saving model...')
    model.save('model')

    ## This is needed for converting from Keras to PyTorch.
    model.save_weights('model.h5')

    #
    # Save model config info.
    #

    print('Saving model.json...')
    ## This is needed for converting from Keras to PyTorch.
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
